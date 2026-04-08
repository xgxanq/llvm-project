//===-- AMDGPUInplaceMFMA.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass adds register allocation hints to MFMA instructions so that the
/// destination register and the accumulator input (Src2) are assigned the same
/// physical register, enabling true in-place accumulation and eliminating
/// unnecessary ACCVGPR copy instructions.
///
/// For regular MFMA instructions the accumulator is operand 3 ($src2).
/// For SMFMAC instructions the accumulator is tied to the destination (operand
/// 0) and its index is determined via the tied-operand mechanism.
///
/// The pass also traces backwards through COPY chains when Src2 is defined by
/// a COPY rather than another MFMA. This helps the allocator coalesce the copy
/// and place the original source in the same register as the MFMA destination,
/// eliminating the copy entirely.
///
/// Reverse hints on Src2 are only set if Src2 does not already carry a hint.
/// This preserves backwards chain propagation in multi-MFMA accumulator chains:
///   %init → %a → %b → %c
/// After processing in program order %a keeps its hint to %init (not %b), so
/// the allocator can follow the chain from any entry point.
///
/// === Chain compression (inspired by AMDGPURewriteAGPRCopyMFMA) ===
///
/// AMDGPURewriteAGPRCopyMFMA (post-RA) identifies the *full* accumulator chain
/// and migrates it as a unit. We apply the same idea pre-RA: after setting
/// pairwise hints we compress every hint chain so that each register points
/// directly to the chain's root (the initialisation register) rather than to
/// its immediate predecessor.
///
/// Before compression (pairwise):
///   %c → %b,  %b → %a,  %a → %init,  %init → %a
///
/// After compression (star topology):
///   %c → %init,  %b → %init,  %a → %init,  %init → %a
///
/// With the star topology the allocator can satisfy every hint regardless of
/// the order in which it processes the virtual registers: once %init is
/// assigned to some physical register P, all other chain members already point
/// directly at P and can be assigned there (subject to non-overlapping live
/// ranges, which canShareReg verifies at each step of the compression).
///
//===----------------------------------------------------------------------===//

#include "AMDGPUInplaceMFMA.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "mfma-in-place-optimizations"

namespace {

class AMDGPUInplaceMFMAImpl {
  const SIInstrInfo    *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;

public:
  AMDGPUInplaceMFMAImpl(LiveIntervals *LIS) : LIS(LIS) {}
  bool run(MachineFunction &MF);

private:
  /// Follow COPY chains up to MaxDepth steps and return the original source
  /// register (the first non-COPY def, or a physical register).
  Register traceCopySrc(Register Reg, unsigned MaxDepth = 4) const;

  /// Return true if A and B can legally share a physical register, i.e.
  /// their live intervals do not overlap.
  bool canShareReg(Register A, Register B) const;

  /// Set a hint on Reg toward Target only if Reg does not already carry a
  /// hint. This prevents later instructions from overwriting hints set by
  /// earlier ones (important for multi-MFMA chains).
  void setHintIfEmpty(Register Reg, Register Target);

  /// Compress hint chains so that every register points directly to the chain
  /// root rather than to its immediate predecessor.  Inspired by
  /// AMDGPURewriteAGPRCopyMFMA's whole-chain identification strategy: that
  /// pass collects the full dst/src2 chain and migrates it as a unit; we do
  /// the equivalent pre-RA by following the hint graph and short-circuiting
  /// each edge to point at the root.
  void compressChainHints() const;
};

class AMDGPUInplaceMFMALegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUInplaceMFMALegacy() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override {
    return "AMDGPU mfma in-place optimizations";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUInplaceMFMALegacy, DEBUG_TYPE,
                      "MFMA in-place optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPUInplaceMFMALegacy, DEBUG_TYPE,
                    "MFMA in-place optimizations", false, false)

char AMDGPUInplaceMFMALegacy::ID = 0;

char &llvm::AMDGPUInplaceMFMAID = AMDGPUInplaceMFMALegacy::ID;

bool AMDGPUInplaceMFMALegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  return AMDGPUInplaceMFMAImpl(LIS).run(MF);
}

PreservedAnalyses
AMDGPUInplaceMFMAPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  AMDGPUInplaceMFMAImpl(LIS).run(MF);
  return PreservedAnalyses::all();
}

// ---------------------------------------------------------------------------
// AMDGPUInplaceMFMAImpl helpers
// ---------------------------------------------------------------------------

Register AMDGPUInplaceMFMAImpl::traceCopySrc(Register Reg,
                                                 unsigned MaxDepth) const {
  for (unsigned D = 0; D < MaxDepth; ++D) {
    if (!Reg.isVirtual())
      break;
    // Use getUniqueVRegDef instead of getVRegDef: kernel arguments and
    // sub-register-initialized tuples may have multiple defs (one per
    // sub-register lane).  getVRegDef asserts uniqueness; getUniqueVRegDef
    // safely returns nullptr in that case, stopping the trace.
    const MachineInstr *DefMI = MRI->getUniqueVRegDef(Reg);
    if (!DefMI || !DefMI->isCopy())
      break;
    Register Src = DefMI->getOperand(1).getReg();
    if (!Src.isVirtual())
      break;
    // Trace through a COPY only when the two register classes share at least
    // one physical register (i.e. have a common sub-class).
    //
    // A strict equality check would stop at any av_*↔vreg_* or av_*↔areg_*
    // boundary introduced by AMDGPUPrepareAGPRAlloc, which widens immediate-
    // initialising AGPR/VGPR writes to av_* so the allocator can freely
    // choose either bank.  A COPY across such a boundary is transparent for
    // hinting: both sides can land in the same physical VGPR (or AGPR).
    if (!TRI->getCommonSubClass(MRI->getRegClass(Reg), MRI->getRegClass(Src)))
      break;
    Reg = Src;
  }
  return Reg;
}

bool AMDGPUInplaceMFMAImpl::canShareReg(Register A, Register B) const {
  if (!LIS)
    return true;
  if (!LIS->hasInterval(A) || !LIS->hasInterval(B))
    return false;
  return !LIS->getInterval(A).overlaps(LIS->getInterval(B));
}

void AMDGPUInplaceMFMAImpl::setHintIfEmpty(Register Reg, Register Target) {
  if (!MRI->getRegAllocationHint(Reg).second.isValid())
    MRI->setRegAllocationHint(Reg, 0, Target);
}

void AMDGPUInplaceMFMAImpl::compressChainHints() const {
  // Walk every virtual register that carries a type-0 hint and follow the
  // hint chain as far as possible, stopping when:
  //   (a) the next hop has no hint, carries a non-standard hint type, or is a
  //       physical register (MRI::getRegAllocationHint asserts on physregs),
  //   (b) a cycle is detected via a visited set,
  //   (c) Reg and the candidate root have overlapping live ranges.
  //
  // PrevRoot tracks the last node entered *before* any exit condition fires,
  // which is the correct compression target:
  //
  //   Example: %c→%b, %b→%a, %a→%init, %init→%a (bidirectional pair at root)
  //   Traversal from %c: visit %b, visit %a, visit %init, try %a → already
  //   visited → exit.  PrevRoot=%init.  Update %c→%init. ✓
  //
  // Without PrevRoot the naive approach would exit with Root=%a (the repeated
  // node), missing the true root %init.
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    auto [HintType, HintReg] = MRI->getRegAllocationHint(Reg);
    if (HintType != 0 || !HintReg.isValid() || !HintReg.isVirtual())
      continue;

    SmallSet<Register, 16> Visited;
    Visited.insert(Reg);

    Register Root = HintReg;
    Register PrevRoot = HintReg;
    while (Visited.insert(Root).second) { // returns false → Root already seen
      auto [NType, NReg] = MRI->getRegAllocationHint(Root);
      if (NType != 0 || !NReg.isValid() || !NReg.isVirtual())
        break; // end of chain or physical-register target
      if (!canShareReg(Reg, NReg))
        break; // live-range interference stops compression here
      PrevRoot = Root;
      Root = NReg;
    }

    // PrevRoot is the last node we successfully entered before the loop exited.
    // Use it as the compression target when it is further than HintReg.
    if (PrevRoot != HintReg) {
      LLVM_DEBUG(dbgs() << "[mfma-inplace] compress "
                        << printReg(Reg, TRI) << " -> "
                        << printReg(HintReg, TRI) << " ==> "
                        << printReg(PrevRoot, TRI) << "\n");
      MRI->setRegAllocationHint(Reg, 0, PrevRoot);
    }
  }
}

// ---------------------------------------------------------------------------
// Main pass logic
// ---------------------------------------------------------------------------

bool AMDGPUInplaceMFMAImpl::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();

  bool Changed = false;

  // Phase 0: promote AGPR-destination MFMAs to VGPR-destination form.
  //
  // By default the instruction selector may emit both AGPR-accumulator
  // (_e64) and VGPR-accumulator (_vgprcd_e64) MFMA variants.  AGPR-dst
  // MFMAs require v_accvgpr_mov_b32 instructions whenever the RA must
  // reposition accumulator values within the AGPR file; they also cause
  // v_accvgpr_write/read pairs whenever an AGPR accumulator interacts with a
  // VGPR-dst MFMA chain.
  //
  // Converting all AGPR-dst MFMAs to their VGPR-dst counterparts (using the
  // getMFMASrcCVDstVGPROp mapping) consolidates all accumulators into the
  // VGPR file before register allocation, eliminating the cross-file moves.
  // The accumulator virtual registers are reclassed from areg_* to vreg_*
  // so the VGPR allocator can assign them together with the rest of the
  // accumulator chain.
  //
  // This transformation is only safe when a VGPR-dst counterpart exists
  // (getMFMASrcCVDstVGPROp returns a valid opcode) and AGPR0 is available
  // (i.e., the target supports the combined AGPR/VGPR register file).
  /*
  if (false && !MRI->isReserved(AMDGPU::AGPR0)) { // Phase 0 disabled for now
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (!SIInstrInfo::isMAI(MI))
          continue;
        int VGPROp = AMDGPU::getMFMASrcCVDstVGPROp(MI.getOpcode());
        if (VGPROp == -1)
          continue; // Already VGPR form or no VGPR counterpart.

        LLVM_DEBUG(dbgs() << "[mfma-inplace] promote to vgprcd: " << MI);
        MI.setDesc(TII->get(VGPROp));

        // Reclass the destination vreg (accumulator output) areg_* → vreg_*.
        Register Dst = MI.getOperand(0).getReg();
        if (Dst.isVirtual()) {
          if (const TargetRegisterClass *VRC =
                  TRI->getEquivalentVGPRClass(MRI->getRegClass(Dst)))
            MRI->setRegClass(Dst, VRC);
        }

        // Reclass the accumulator input (src2) vreg.  For non-tied MFMAs
        // this is a separate virtual register; for tied (_mac_*) variants it
        // is the same vreg as Dst and is already updated above.
        unsigned AccIdx = 3;
        unsigned TiedIdx;
        if (MI.isRegTiedToUseOperand(0, &TiedIdx))
          AccIdx = TiedIdx;
        const MachineOperand &AccMO = MI.getOperand(AccIdx);
        if (AccMO.isReg() && AccMO.getReg().isVirtual() &&
            AccMO.getReg() != Dst) {
          Register Src2 = AccMO.getReg();
          if (const TargetRegisterClass *VRC =
                  TRI->getEquivalentVGPRClass(MRI->getRegClass(Src2)))
            MRI->setRegClass(Src2, VRC);
        }

        Changed = true;
      }
    }
  }
  */
  // Phase 1: set pairwise hints — each MFMA destination is hinted toward its
  // accumulator source (possibly tracing back through COPY chains).
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (!SIInstrInfo::isMFMA(MI))
        continue;

      Register Dst = MI.getOperand(0).getReg();
      if (!Dst.isVirtual())
        continue;

      // For SMFMAC the accumulator ($src2) is tied to the destination and sits
      // at a different index than regular MFMA. Use the tied-operand mechanism
      // to locate it generically; fall back to operand 3 for non-tied MFMA.
      unsigned AccIdx = 3;
      unsigned TiedIdx;
      if (MI.isRegTiedToUseOperand(0, &TiedIdx))
        AccIdx = TiedIdx;

      const MachineOperand &AccMO = MI.getOperand(AccIdx);
      if (!AccMO.isReg())
        continue;
      Register Src2 = AccMO.getReg();
      if (!Src2.isVirtual())
        continue;

      // Trace through any COPY chain so we hint toward the original source.
      // This allows the allocator to coalesce the COPY and place the original
      // accumulator value directly in the MFMA destination register.
      Register AccSrc = traceCopySrc(Src2);

      // Skip if the live ranges overlap; an impossible hint may mislead the
      // allocator into making worse choices elsewhere.
      if (!canShareReg(Dst, AccSrc))
        continue;

      LLVM_DEBUG(dbgs() << "[mfma-inplace] hinting " << MI);

      // Primary hint: allocate Dst to the same register as the original
      // accumulator source (possibly reaching through a COPY chain).
      MRI->setRegAllocationHint(Dst, 0, AccSrc);

      // Reverse hint on AccSrc: only set if AccSrc has no prior hint.
      // This preserves backward chain propagation — in a sequence
      //   %init → %a → %b → %c
      // %a's hint (→ %init) is set when processing the first MFMA and must
      // not be overwritten by the reverse hint from the second MFMA (→ %b).
      setHintIfEmpty(AccSrc, Dst);

      // If there was a COPY chain (AccSrc != Src2), also nudge the
      // intermediate copy registers toward Dst so the allocator can collapse
      // the entire chain into a single register.
      if (AccSrc != Src2)
        setHintIfEmpty(Src2, Dst);

      Changed = true;
    }
  }

  if (!Changed)
    return false;

  // Phase 2: compress hint chains toward their roots.
  //
  // After Phase 1 the hints form a linear backward chain:
  //   %c → %b,  %b → %a,  %a → %init,  %init → %a
  //
  // The allocator only follows one hint level deep, so if it allocates %c
  // before %b is assigned, the hint %c → %b is unresolvable and %c may land
  // in a different physical register than the rest of the chain.
  //
  // Compression rewrites each intermediate register to point directly at the
  // root, creating a star topology:
  //   %c → %init,  %b → %init,  %a → %init,  %init → %a
  //
  // Now every chain member's hint is independently satisfiable once %init is
  // assigned, regardless of allocation order — the same guarantee that
  // AMDGPURewriteAGPRCopyMFMA achieves post-RA by migrating the entire chain
  // as an atomic unit.
  compressChainHints();

  return true;
}
