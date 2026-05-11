//===-- AMDGPUConvertMFMAVGPRToAGPR.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Rewrite V_MFMA_F32_16X16X32_F16_vgprcd_e64 instructions inside innermost
/// loops to V_MFMA_F32_16X16X32_F16_e64, converting the C (src2) and D (vdst)
/// virtual register classes from VGPR to equivalent AGPR classes.
///
/// Accumulator registers that escape the loop (used by instructions outside
/// the loop body) are handled by inserting AGPR→VGPR copy instructions at the
/// loop exit, provided every outside use accepts a VGPR operand.
///
//===----------------------------------------------------------------------===//

#include "AMDGPUConvertMFMAVGPRToAGPR.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-convert-mfma-vgpr-to-agpr"

static cl::opt<bool> EnableMFMAVGPRToAGPR(
    "amdgpu-enable-mfma-vgpr-to-agpr",
    cl::desc("Enable rewriting vgprcd MFMA instructions to AGPR form in "
             "innermost loops"),
    cl::init(true), cl::Hidden);

static cl::opt<unsigned> MFMAVGPRToAGPRCommitPercent(
    "amdgpu-mfma-vgpr-to-agpr-commit-percent",
    cl::desc("Percentage (0-100) of valid connected components to rewrite from "
             "vgprcd MFMA to AGPR form per innermost loop (default: 50)"),
    cl::init(50), cl::Hidden);

STATISTIC(NumMFMAsRewritten,
          "Number of vgprcd MFMA instructions rewritten to AGPR form");
STATISTIC(NumCopiesInserted,
          "Number of AGPR->VGPR copies inserted at loop exits");

namespace {

/// A use of an accumulator register by an instruction outside all loops.
/// Fixed by inserting an AGPR→VGPR copy at one of three points depending on
/// context: just after the def (PHI use with def outside all loops), before the
/// predecessor BB's terminator (PHI use with def inside a loop), or at
/// getFirstNonPHI() of the user's BB (non-PHI use).
struct EscapeUse {
  Register Reg;       ///< The accumulator register being escaped.
  MachineInstr *User; ///< Instruction outside all loops that uses it.
  unsigned OpNo;      ///< Operand index in User.
};

class AMDGPUConvertMFMAVGPRToAGPRImpl {
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  MachineRegisterInfo &MRI;
  const MachineLoopInfo &MLI;
  SlotIndexes &Indexes;

  /// Cache for opcode -> isVgprcdCandidate results to avoid repeated
  /// binary-search lookups into the TableGen-generated mapping table.
  mutable DenseMap<unsigned, bool> IsVgprcdCache;

  bool isVgprcdCandidate(const MachineInstr &MI) const {
    unsigned Opc = MI.getOpcode();
    auto [It, Inserted] = IsVgprcdCache.try_emplace(Opc);
    if (Inserted)
      It->second = AMDGPU::getMFMASrcCVDstAGPROp(Opc) != -1;
    return It->second;
  }

  /// Returns true if \p Def (which defines \p Reg) can output an AGPR-class
  /// value.  vgprcd MFMAs are treated as AGPR-producible because this pass
  /// will rewrite them.  Results are cached to avoid repeated lookups.
  bool defCanProduceAGPR(MachineInstr &Def, Register Reg,
                         DenseMap<MachineInstr *, bool> &Cache) const {
    // collectComponent's bidirectional DFS guarantees that if MFMA_B.vdst ==
    // Src2Reg == MFMA_A.src2, both MFMAs end up in the same component
    // (DefInComp = true → case (b), this function is never called).
    // Reaching here with a vgprcd Def means the DFS invariant is broken.
    assert(!isVgprcdCandidate(Def) &&
           "vgprcd Src2Def reached defCanProduceAGPR with DefInComp=false "
           "— DFS chain-following invariant violated");
    auto [It, Inserted] = Cache.try_emplace(&Def, true);
    if (!Inserted)
      return It->second;
    for (unsigned I = 0, E = Def.getNumOperands(); I != E; ++I) {
      const MachineOperand &MO = Def.getOperand(I);
      if (!MO.isReg() || !MO.isDef() || MO.getReg() != Reg)
        continue;
      const TargetRegisterClass *C = Def.getRegClassConstraint(I, &TII, &TRI);
      It->second = !C || TRI.hasAGPRs(C);
      break;
    }
    return It->second;
  }

  /// After \p Reg has been promoted to \p EquivRC (AGPR class), insert one
  /// AGPR→VGPR copy per use-BB for the VGPR-only escape uses recorded in
  /// \p EscapesByReg[Reg], and redirect each such use to the copy result.
  void fixVGPROnlyEscapes(
      Register Reg, const TargetRegisterClass *EquivRC,
      const DenseMap<Register, SmallVector<EscapeUse, 4>> &EscapesByReg) {
    const TargetRegisterClass *VGPRClass =
        TRI.getEquivalentVGPRClass(EquivRC);
    assert(VGPRClass && "AGPR class has no equivalent VGPR class");
    auto RegIt = EscapesByReg.find(Reg);
    if (RegIt == EscapesByReg.end())
      return;
    SmallVector<MachineOperand *, 8> VGPROnlyUses;
    SmallVector<MachineOperand *, 8> PHIVGPROnlyUses;
    for (const EscapeUse &EU : RegIt->second) {
      MachineInstr *User = EU.User;
      if (User->isPHI()) {
        // getRegClassConstraint returns null for PHI incoming value operands.
        // Check the PHI result's RC instead.
        Register PhiResult = User->getOperand(0).getReg();
        if (!TRI.hasAGPRs(MRI.getRegClass(PhiResult)))
          PHIVGPROnlyUses.push_back(&User->getOperand(EU.OpNo));
      } else {
        const TargetRegisterClass *C =
            User->getRegClassConstraint(EU.OpNo, &TII, &TRI);
        if (C && !TRI.hasAGPRs(C))
          VGPROnlyUses.push_back(&User->getOperand(EU.OpNo));
      }
    }
    if (VGPROnlyUses.empty() && PHIVGPROnlyUses.empty())
      return;
    // Shared dedup map: one AGPR→VGPR copy per unique InsertBB, reused by
    // both non-PHI and PHI escape uses.
    MachineInstr *Def = MRI.getVRegDef(Reg);
    DenseMap<MachineBasicBlock *, Register> CopyPerInsertBB;
    auto emitCopy = [&](MachineOperand *MO, MachineBasicBlock *InsertBB,
                        MachineBasicBlock::iterator InsertPt) {
      auto [It, Inserted] = CopyPerInsertBB.try_emplace(InsertBB);
      if (Inserted) {
        It->second = MRI.createVirtualRegister(VGPRClass);
        BuildMI(*InsertBB, InsertPt, DebugLoc(),
                TII.get(TargetOpcode::COPY), It->second)
            .addReg(Reg);
        ++NumCopiesInserted;
        LLVM_DEBUG(dbgs() << "[commitComponent] VGPR-only escape: COPY "
                          << printReg(It->second, &TRI) << " = "
                          << printReg(Reg, &TRI) << " in BB#"
                          << InsertBB->getNumber() << "\n");
      }
      MO->setReg(It->second);
    };

    // Non-PHI uses: insert copy after the def if the use is in the def's BB
    // (an out-of-loop MFMA in the component), otherwise at getFirstNonPHI().
    for (MachineOperand *MO : VGPROnlyUses) {
      MachineBasicBlock *InsertBB = MO->getParent()->getParent();
      MachineBasicBlock::iterator InsertPt =
          (Def->getParent() == InsertBB)
              ? std::next(Def->getIterator())
              : InsertBB->getFirstNonPHI();
      emitCopy(MO, InsertBB, InsertPt);
    }

    // PHI uses: if Reg's def is outside all loops, hoist the copy to just
    // after the def (executed once).  Otherwise insert before PredBB's
    // terminator (on the exit edge, inside the loop — unavoidable).
    // PHI operand layout: val0, bb0, val1, bb1, ... — pred MBB is at OpNo+1.
    bool DefIsOutsideLoop = !MLI.getLoopFor(Def->getParent());
    for (MachineOperand *MO : PHIVGPROnlyUses) {
      MachineInstr *PHI = MO->getParent();
      unsigned OpIdx = MO->getOperandNo();
      MachineBasicBlock *PredBB = PHI->getOperand(OpIdx + 1).getMBB();
      MachineBasicBlock *InsertBB;
      MachineBasicBlock::iterator InsertPt;
      if (DefIsOutsideLoop) {
        InsertBB = Def->getParent();
        InsertPt = std::next(Def->getIterator());
      } else {
        assert(PredBB->getFirstTerminator() != PredBB->end() &&
               "loop exit predecessor must have a terminator");
        InsertBB = PredBB;
        InsertPt = PredBB->getFirstTerminator();
      }
      emitCopy(MO, InsertBB, InsertPt);
    }
  }

  /// Collect the cross-loop connected component of vgprcd MFMAs reachable from
  /// \p Root via accumulator-chain edges (dst→src2 in both directions),
  /// crossing any loop boundary.  All reachable vgprcd MFMAs go into \p
  /// Component.
  ///
  /// The DFS uses accumulator registers as worklist nodes (one visited set,
  /// no dual MI+Register tracking).  The caller is responsible for adding all
  /// Component members to the global Visited set after this call returns.
  ///
  /// Non-chain uses of accumulator registers are classified by loop membership:
  ///   - Inside any loop: recorded in \p PendingCompatChecks.
  ///   - Outside all loops: recorded in \p Escapes.
  void collectComponent(
      MachineInstr &Root,
      SetVector<MachineInstr *> &Component,
      SmallVectorImpl<EscapeUse> &Escapes,
      SmallVectorImpl<const MachineOperand *> &PendingCompatChecks) {
    // AccumRegs: DFS visited set at register level — prevents rescanning the
    // same register's use-list.  Local; not exposed to callers.
    DenseSet<Register> AccumRegs;
    SmallVector<Register, 16> Worklist;

    // Add MI to Component exactly once and seed its two accumulator registers.
    // MIs whose src2 is not a virtual register cannot be rewritten to AGPR
    // form and are excluded from Component.
    // SetVector::insert returns false if already present — handles dedup.
    auto tryAddChainMember = [&](MachineInstr *MI) {
      MachineOperand *Src2MO = TII.getNamedOperand(*MI, AMDGPU::OpName::src2);
      if (!Src2MO)
        return;
      if (!Component.insert(MI))
        return;
      LLVM_DEBUG(dbgs() << "[collectComponent] Member: " << *MI);
      MachineOperand *VDstMO = TII.getNamedOperand(*MI, AMDGPU::OpName::vdst);
      if (AccumRegs.insert(VDstMO->getReg()).second)
        Worklist.push_back(VDstMO->getReg());
      // Only follow src2 backward when it is a virtual register.
      if (Src2MO->isReg() && Src2MO->getReg().isVirtual())
        if (AccumRegs.insert(Src2MO->getReg()).second)
          Worklist.push_back(Src2MO->getReg());
    };

    // Seed DFS from Root.
    tryAddChainMember(&Root);

    while (!Worklist.empty()) {
      Register Reg = Worklist.pop_back_val();

      // Forward edges: find downstream MFMAs that consume Reg as src2.
      for (const MachineOperand &MO : MRI.use_nodbg_operands(Reg)) {
        MachineInstr *User = const_cast<MachineInstr *>(MO.getParent());

        if (isVgprcdCandidate(*User)) {
          MachineOperand *USrc2 =
              TII.getNamedOperand(*User, AMDGPU::OpName::src2);
          if (USrc2 && USrc2->isReg() && USrc2->getReg() == Reg) {
            // Accumulator-chain edge: User consumes Reg as src2.
            tryAddChainMember(User);
            continue;
          }
          // vgprcd use via non-accumulator operand (e.g. src0/src1).
        }

        // Non-chain use: classify by loop membership.
        if (MLI.getLoopFor(User->getParent())) {
          // Inside a loop — inserting a copy here would add per-iteration
          // overhead.  Require AGPR compatibility; reject if not.
          // Include subreg operands: getRegClassConstraint on a subreg slot
          // correctly reflects whether the slot accepts AGPR after the vreg
          // class is changed, so skipping them would leave RC conflicts
          // undetected.
          PendingCompatChecks.push_back(&MO);
        } else {
          // Outside all loops — fixable by an AGPR→VGPR copy.
          EscapeUse EU;
          EU.Reg = Reg;
          EU.User = User;
          EU.OpNo = MO.getOperandNo();
          Escapes.push_back(EU);
        }
      }

      // Backward edge: follow def unconditionally across all loops.
      // getVRegDef(Reg)->vdst == Reg (SSA), so only Src2 of Def is new.
      MachineInstr *Def = MRI.getVRegDef(Reg);
      assert(Def && "AccumReg has no def — MIR is not in SSA form");
      if (isVgprcdCandidate(*Def)) {
        tryAddChainMember(Def);
      } else if (MLI.getLoopFor(Def->getParent())) {
        // Non-vgprcd def inside a loop: record def operand for compat
        // check.  SSA guarantees a unique def, so def_operands yields
        // exactly one operand.
        for (const MachineOperand &DefMO : MRI.def_operands(Reg))
          PendingCompatChecks.push_back(&DefMO);
      }
    }
  }

  /// Check that all in-loop non-chain operands collected during DFS are
  /// compatible with AGPR.  Component members are skipped because they will
  /// be rewritten.  Both full-register and subreg operands are checked.
  bool checkPendingCompat(
      const SmallVectorImpl<const MachineOperand *> &PendingCompatChecks,
      const SetVector<MachineInstr *> &CompSet) const {
    for (const MachineOperand *MO : PendingCompatChecks) {
      const MachineInstr *MI = MO->getParent();
      if (CompSet.count(const_cast<MachineInstr *>(MI)))
        continue; // In component — will be rewritten to AGPR.
      unsigned OpNo = MO->getOperandNo();
      const TargetRegisterClass *Constraint =
          MI->getRegClassConstraint(OpNo, &TII, &TRI);
      if (!Constraint) {
        LLVM_DEBUG(dbgs() << "[checkPendingCompat] op#" << OpNo
                          << " no RC constraint (ok): " << *MI);
        continue;
      }
      LLVM_DEBUG(dbgs() << "[checkPendingCompat] op#" << OpNo
                        << " constraint=" << TRI.getRegClassName(Constraint)
                        << " hasAGPRs=" << TRI.hasAGPRs(Constraint) << ": "
                        << *MI);
      if (!TRI.hasAGPRs(Constraint)) {
        LLVM_DEBUG(dbgs() << "[checkPendingCompat] REJECT: op#" << OpNo
                          << " needs VGPR-only class "
                          << TRI.getRegClassName(Constraint)
                          << ", cannot convert to AGPR.\n");
        return false;
      }
    }
    return true;
  }

  /// Check that every escape use can be satisfied by inserting an AGPR→VGPR
  /// copy (i.e. the outside user's constraint is VGPR-only or unconstrained).
  bool escapesFixableByCopy(const SmallVectorImpl<EscapeUse> &Escapes) const {
    for (const EscapeUse &EU : Escapes) {
      const TargetRegisterClass *Constraint =
          EU.User->getRegClassConstraint(EU.OpNo, &TII, &TRI);
      if (!Constraint) {
        // No constraint: a VGPR copy is always acceptable.
        continue;
      }
      if (TRI.hasAGPRs(Constraint)) {
        // Already accepts AGPR directly — no copy needed, still fine.
        continue;
      }
      if (TRI.hasVGPRs(Constraint)) {
        // VGPR-only: fixable by inserting an AGPR→VGPR copy.
        LLVM_DEBUG(dbgs() << "[escapesFixableByCopy] Reg "
                          << printReg(EU.Reg, &TRI) << " op#" << EU.OpNo
                          << " constraint=" << TRI.getRegClassName(Constraint)
                          << " → fixable by copy: " << *EU.User);
        continue;
      }
      // Constraint accepts neither VGPRs nor AGPRs: cannot fix.
      LLVM_DEBUG(dbgs() << "[escapesFixableByCopy] REJECT: Reg "
                        << printReg(EU.Reg, &TRI) << " op#" << EU.OpNo
                        << " constraint=" << TRI.getRegClassName(Constraint)
                        << " is not VGPR/AGPR: " << *EU.User);
      return false;
    }
    return true;
  }

  /// Commit the rewrite for the unified connected component.
  ///
  /// \p Component contains all vgprcd MFMAs in the accumulator chain — both
  /// loop-internal and loop-external chain members — in DFS insertion order.
  /// The rewrite is order-independent: cases (b)/(c)/(d) handle src2 correctly
  /// regardless of whether the upstream MFMA has been processed yet.
  ///
  /// For each MFMA:
  ///   1. Convert src2 RC and dst RC to their AGPR equivalents.
  ///   2. Replace opcode with the AGPR form.
  ///   3. For any use of the (now AGPR) dst that does not accept AGPR, insert
  ///      an AGPR→VGPR copy immediately after the MFMA and redirect that use.
  ///      Chain successors (also in Component, to be rewritten) are excluded.
  void commitComponent(const SetVector<MachineInstr *> &Component,
                       const SmallVectorImpl<EscapeUse> &Escapes) {
    DenseMap<MachineInstr *, bool> DefCanBeAGPRCache;
    // Keyed by Src2Reg so multi-def instructions get independent copies.
    DenseMap<Register, Register> Src2ToAGPRCopyReg;
    // Prevent redundant fixVGPROnlyEscapes calls for shared Src2Regs.
    DenseSet<Register> FixedSrc2Regs;

    // Pre-index Escapes by register for O(1) lookup in fixVGPROnlyEscapes,
    // replacing the previous O(E) linear scan per call.
    DenseMap<Register, SmallVector<EscapeUse, 4>> EscapesByReg;
    for (const EscapeUse &EU : Escapes)
      EscapesByReg[EU.Reg].push_back(EU);

    for (MachineInstr *MI : Component) {
      int NewOpc = AMDGPU::getMFMASrcCVDstAGPROp(MI->getOpcode());
      assert(NewOpc != -1 && "Component member must be a vgprcd candidate");

      MachineOperand *Src2MO = TII.getNamedOperand(*MI, AMDGPU::OpName::src2);
      Register DstReg = TII.getNamedOperand(*MI, AMDGPU::OpName::vdst)->getReg();
      bool Src2IsVReg = Src2MO && Src2MO->isReg() && Src2MO->getReg().isVirtual();
      Register Src2Reg = Src2IsVReg ? Src2MO->getReg() : Register();
      const TargetRegisterClass *EquivDst =
          TRI.getEquivalentAGPRClass(MRI.getRegClass(DstReg));
      const TargetRegisterClass *EquivSrc2 =
          Src2IsVReg ? TRI.getEquivalentAGPRClass(MRI.getRegClass(Src2Reg))
                     : nullptr;

      assert(EquivDst && "vgprcd MFMA vdst has no AGPR equivalent class");
      assert((!Src2IsVReg || EquivSrc2) &&
             "vgprcd MFMA src2 has no AGPR equivalent class");

      // ── Rewrite src2 ────────────────────────────────────────────────────
      // (a) Src2Reg already AGPR → nothing to do; escape uses of Src2Reg were
      //     already fixed when the upstream MFMA that produced it was processed
      //     (setRegClass + fixVGPROnlyEscapes in an earlier iteration of this
      //     loop, or Src2Reg was naturally AGPR before this pass).
      // (b) Src2Def in component → setRegClass (its vdst will also be AGPR).
      // (c) Src2Def outside, can produce AGPR → setRegClass + fix escapes.
      // (d) Src2Def outside, VGPR-only → insert VGPR→AGPR copy after def.
      // src2 may be an immediate (no-op: no register to rewrite).
      if (Src2IsVReg && !TRI.hasAGPRs(MRI.getRegClass(Src2Reg))) {
        MachineInstr *Src2Def = MRI.getVRegDef(Src2Reg);
        bool DefInComp = Component.count(Src2Def);
        if (DefInComp || defCanProduceAGPR(*Src2Def, Src2Reg,
                                          DefCanBeAGPRCache)) {
          // Cases (b) and (c).
          MRI.setRegClass(Src2Reg, EquivSrc2);
          if (!DefInComp && FixedSrc2Regs.insert(Src2Reg).second)
            fixVGPROnlyEscapes(Src2Reg, EquivSrc2, EscapesByReg);
        } else {
          // Case (d): Src2Def is VGPR-only and outside all loops
          // (checkPendingCompat rejects in-loop VGPR-only defs).
          assert(!MLI.getLoopFor(Src2Def->getParent()) &&
                 "case(d): Src2Def inside a loop — checkPendingCompat "
                 "should have rejected this component");
          auto [It, Inserted] = Src2ToAGPRCopyReg.try_emplace(Src2Reg);
          if (Inserted) {
            Register AGPRSrc2 = MRI.createVirtualRegister(EquivSrc2);
            MachineBasicBlock *InsertBB = Src2Def->getParent();
            auto InsertPt = std::next(Src2Def->getIterator());
            BuildMI(*InsertBB, InsertPt, DebugLoc(),
                    TII.get(TargetOpcode::COPY), AGPRSrc2)
                .addReg(Src2Reg);
            It->second = AGPRSrc2;
            ++NumCopiesInserted;
            LLVM_DEBUG(dbgs()
                       << "[commitComponent] case(d): COPY "
                       << printReg(AGPRSrc2, &TRI) << " = "
                       << printReg(Src2Reg, &TRI) << " after Src2Def\n");
          }
          Src2MO->setReg(It->second);
        }
      }

      LLVM_DEBUG(dbgs() << "[commitComponent] Before rewrite: " << *MI);
      MRI.setRegClass(DstReg, EquivDst);
      MI->setDesc(TII.get(NewOpc));
      LLVM_DEBUG(dbgs() << "[commitComponent] After  rewrite: " << *MI);
      ++NumMFMAsRewritten;

      fixVGPROnlyEscapes(DstReg, EquivDst, EscapesByReg);
    }
  }

public:
  AMDGPUConvertMFMAVGPRToAGPRImpl(const GCNSubtarget &ST,
                                 MachineRegisterInfo &MRI,
                                 const MachineLoopInfo &MLI,
                                 SlotIndexes &Indexes)
      : TII(*ST.getInstrInfo()), TRI(*ST.getRegisterInfo()), MRI(MRI),
        MLI(MLI), Indexes(Indexes) {}

  bool run(MachineFunction &MF) {
    if (!EnableMFMAVGPRToAGPR)
      return false;
    const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
    if (!ST.hasGFX90AInsts())
      return false;
    if (MRI.isReserved(AMDGPU::AGPR0))
      return false;

    bool Changed = false;

    LLVM_DEBUG(dbgs() << "[AMDGPUConvertMFMAVGPRToAGPR] Running on: "
                      << MF.getName() << "\n");
    LLVM_DEBUG(
        dbgs() << "[AMDGPUConvertMFMAVGPRToAGPR] MachineFunction before pass:\n";
        MF.print(dbgs()));

    // Global visited set: each vgprcd MI is processed exactly once.
    DenseSet<MachineInstr *> Visited;

    // Walk all innermost loops in BB layout order (linear MF scan).
    // Processing in program order ensures that when a cross-loop accumulator
    // chain exists (e.g. L1 feeds L2 feeds L3), the root MFMA chosen by DFS
    // is in L1 (the earliest loop), so all chain MFMAs across L1/L2/L3 land
    // in a single Component rooted at L1.  If L2 were scanned first, the same
    // Component would be rooted at L2, and the L1 chain members would still be
    // collected via the backward edge — but the DFS would process L2's
    // registers before L1's, potentially triggering case (d) copies that a
    // later iteration of the commitComponent loop renders redundant.
    // Each loop's header appears before its blocks in layout order, so this
    // naturally yields program order without an explicit reverse step.
    SmallVector<MachineLoop *, 16> InnermostLoops;
    DenseSet<MachineLoop *> SeenLoops;
    for (MachineBasicBlock &MBB : MF) {
      MachineLoop *L = MLI.getLoopFor(&MBB);
      if (L && L->isInnermost() && SeenLoops.insert(L).second)
        InnermostLoops.push_back(L);
    }

    // Global collection: scan all innermost loops and build cross-loop
    // connected components.  A single component may span multiple loops.
    struct ComponentData {
      SetVector<MachineInstr *> Component; // all MFMAs in the chain, ordered + O(1) lookup
      SmallVector<EscapeUse, 16> Escapes;
      SmallVector<const MachineOperand *, 16> PendingCompatChecks;
    };
    SmallVector<ComponentData, 8> ValidComponents;

    for (MachineLoop *L : InnermostLoops) {
      for (MachineBasicBlock *MBB : L->getBlocks()) {
        for (MachineInstr &MI : *MBB) {
          if (!isVgprcdCandidate(MI))
            continue;
          if (Visited.count(&MI))
            continue;

          ComponentData CD;

          LLVM_DEBUG(dbgs() << "[run] Candidate MFMA in BB#"
                            << MBB->getNumber() << ": " << MI);

          collectComponent(MI, CD.Component, CD.Escapes,
                           CD.PendingCompatChecks);

          // Mark all component members globally so they are not re-rooted.
          Visited.insert(CD.Component.begin(), CD.Component.end());

          LLVM_DEBUG(dbgs() << "[run] Component: " << CD.Component.size()
                            << " MFMAs (cross-loop), " << CD.Escapes.size()
                            << " escape uses.\n");

          // Reject if any in-loop non-chain use is AGPR-incompatible.
          if (!checkPendingCompat(CD.PendingCompatChecks, CD.Component)) {
            LLVM_DEBUG(dbgs() << "[run] REJECTED: in-loop use incompatible "
                                 "with AGPR.\n");
            continue;
          }

          // Reject if any out-of-loop escape use cannot be fixed by copy.
          if (!escapesFixableByCopy(CD.Escapes)) {
            LLVM_DEBUG(dbgs() << "[run] REJECTED: escape use not fixable by "
                                 "copy.\n");
            continue;
          }

          if (CD.Component.empty())
            continue;
          LLVM_DEBUG(dbgs() << "[run] ACCEPTED.\n");
          ValidComponents.push_back(std::move(CD));
        }
      }
    }

    if (ValidComponents.empty())
      return false;

    unsigned AGPRBudget = ST.getMaxNumVectorRegs(MF.getFunction()).second;

    LLVM_DEBUG(dbgs() << "[run] AGPRBudget=" << AGPRBudget << "\n");

    // Build per-component full SlotIndex intervals spanning ALL MFMAs in the
    // component (including out-of-loop chain members).
    struct CompInterval {
      SlotIndex Start, End;
      unsigned VGPRSlots = 0;
      bool Valid = false;
    };
    SmallVector<CompInterval, 8> LoopIntervals(ValidComponents.size());
    for (size_t I = 0; I < ValidComponents.size(); ++I) {
      CompInterval &LI = LoopIntervals[I];
      for (const MachineInstr *MI : ValidComponents[I].Component) {
        Register Dst =
            TII.getNamedOperand(*MI, AMDGPU::OpName::vdst)->getReg();
        SlotIndex S = Indexes.getInstructionIndex(*MI);
        if (!LI.Valid) {
          LI.Start = LI.End = S;
          LI.VGPRSlots = TRI.getSpillSize(*MRI.getRegClass(Dst)) / 4;
          LI.Valid = true;
        } else {
          if (S < LI.Start) LI.Start = S;
          if (S > LI.End)   LI.End   = S;
        }
      }
    }

    // Sweep-line to find the peak overlap SlotIndex.
    using SweepEvent = std::pair<SlotIndex, int>;
    SmallVector<SweepEvent, 16> SweepEvents;
    for (size_t I = 0; I < ValidComponents.size(); ++I) {
      if (!LoopIntervals[I].Valid)
        continue;
      int W = (int)LoopIntervals[I].VGPRSlots;
      SweepEvents.push_back({LoopIntervals[I].Start, +W});
      SweepEvents.push_back({LoopIntervals[I].End.getNextSlot(), -W});
    }
    llvm::sort(SweepEvents, [](const SweepEvent &A, const SweepEvent &B) {
      if (A.first != B.first)
        return A.first < B.first;
      return A.second < B.second; // -W before +W at same slot
    });
    SlotIndex PeakSlot;
    unsigned PeakPressure = 0;
    {
      int Cur = 0;
      for (const auto &[Slot, Delta] : SweepEvents) {
        Cur += Delta;
        if ((unsigned)Cur > PeakPressure) {
          PeakPressure = (unsigned)Cur;
          PeakSlot = Slot;
        }
      }
    }

    LLVM_DEBUG(dbgs() << "[run] PeakPressure=" << PeakPressure
                      << " PeakSlot=" << PeakSlot << "\n");

    // Collect all components whose interval contains PeakSlot, in program order.
    SmallVector<size_t, 16> PeakIndices;
    if (PeakSlot.isValid()) {
      for (size_t I = 0; I < ValidComponents.size(); ++I) {
        const CompInterval &LI = LoopIntervals[I];
        if (!LI.Valid)
          continue;
        if (LI.Start <= PeakSlot && PeakSlot <= LI.End)
          PeakIndices.push_back(I);
      }
    }
    // If MFMAVGPRToAGPRCommitPercent is non-zero, use it as an independent
    // target count.  When the target exceeds PeakIndices, fill in additional
    // components from ValidComponents in program order (skipping duplicates).
    // When the target is less than PeakIndices, keep all of PeakIndices.
    // The greedy budget trim below enforces the hard AGPR budget cap.
    if (MFMAVGPRToAGPRCommitPercent > 0) {
      unsigned Pct = std::min(MFMAVGPRToAGPRCommitPercent.getValue(), 100u);
      size_t Target = (ValidComponents.size() * Pct + 99) / 100;
      if (Target > PeakIndices.size()) {
        DenseSet<size_t> InPeak(PeakIndices.begin(), PeakIndices.end());
        for (size_t I = 0; I < ValidComponents.size() && PeakIndices.size() < Target; ++I) {
          if (!InPeak.count(I))
            PeakIndices.push_back(I);
        }
      }
      // If Target <= PeakIndices.size(), keep all PeakIndices unchanged.
    }

    // Greedy AGPR budget trim: accumulate each component's actual slot count
    // and stop before exceeding the per-wave AGPR budget.  Using per-component
    // VGPRSlots (from CompInterval) handles mixed-size MFMA functions correctly.
    {
      unsigned AccumSlots = 0;
      size_t CommitCount = 0;
      for (size_t Idx : PeakIndices) {
        unsigned Slots = LoopIntervals[Idx].VGPRSlots;
        if (AccumSlots + Slots > AGPRBudget)
          break;
        AccumSlots += Slots;
        ++CommitCount;
      }
      PeakIndices.resize(CommitCount);
    }

    LLVM_DEBUG(dbgs() << "[run] PeakComponents=" << PeakIndices.size() << "\n");

    for (size_t I : PeakIndices) {
      ComponentData &CD = ValidComponents[I];
      LLVM_DEBUG(dbgs() << "[run] Committing component #" << I << " ("
                        << CD.Component.size() << " MFMAs).\n");
      commitComponent(CD.Component, CD.Escapes);
      Changed = true;
    }

    return Changed;
  }
};

// ---------------------------------------------------------------------------
// Legacy pass wrapper
// ---------------------------------------------------------------------------

class AMDGPUConvertMFMAVGPRToAGPRLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUConvertMFMAVGPRToAGPRLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AMDGPU Rewrite MFMA vgprcd to AGPR in Innermost Loops";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<SlotIndexesWrapperPass>();
    AU.setPreservesCFG();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
    MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    SlotIndexes &Indexes = getAnalysis<SlotIndexesWrapperPass>().getSI();
    return AMDGPUConvertMFMAVGPRToAGPRImpl(ST, MF.getRegInfo(), MLI, Indexes)
        .run(MF);
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(AMDGPUConvertMFMAVGPRToAGPRLegacy, DEBUG_TYPE,
                      "AMDGPU Rewrite MFMA vgprcd to AGPR in Innermost Loops",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_END(AMDGPUConvertMFMAVGPRToAGPRLegacy, DEBUG_TYPE,
                    "AMDGPU Rewrite MFMA vgprcd to AGPR in Innermost Loops",
                    false, false)

char AMDGPUConvertMFMAVGPRToAGPRLegacy::ID = 0;
char &llvm::AMDGPUConvertMFMAVGPRToAGPRLegacyID =
    AMDGPUConvertMFMAVGPRToAGPRLegacy::ID;

// ---------------------------------------------------------------------------
// New pass manager wrapper
// ---------------------------------------------------------------------------

PreservedAnalyses
AMDGPUConvertMFMAVGPRToAGPRPass::run(MachineFunction &MF,
                                    MachineFunctionAnalysisManager &MFAM) {
  MachineLoopInfo &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  SlotIndexes &Indexes = MFAM.getResult<SlotIndexesAnalysis>(MF);
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  AMDGPUConvertMFMAVGPRToAGPRImpl Impl(ST, MF.getRegInfo(), MLI, Indexes);
  if (!Impl.run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<MachineLoopAnalysis>();
  return PA;
}
