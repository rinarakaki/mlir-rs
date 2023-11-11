/*!
Target triple helper class

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/TargetParser/Triple.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/TargetParser/Triple.cpp>
*/

#![allow(non_camel_case_types)]

/**
Triple - Helper class for working with autoconf configuration names. For historical reasons, we also call these 'triples' (they used to contain exactly three fields).

Configuration names are strings in the canonical form:
  ARCHITECTURE-VENDOR-OPERATING_SYSTEM
or
  ARCHITECTURE-VENDOR-OPERATING_SYSTEM-ENVIRONMENT

This class is used for clients which want to support arbitrary configuration names, but also want to implement certain special behaviour for particular configurations. This class isolates the mapping from the components of the configuration name to well known IDs.

At its core the Triple class is designed to be a wrapper for a triple string; the constructor does not change or normalize the triple string.
Clients that need to handle the non-canonical triples that users often specify should use the normalize method.

See autoconf/config.guess for a glimpse into what configuration names look like in practice.
*/
pub struct Triple {
}

pub enum ArchType {
    UnknownArch,

    arm,            // ARM (little endian): arm, armv.*, xscale
    armeb,          // ARM (big endian): armeb
    aarch64,        // AArch64 (little endian): aarch64
    aarch64_be,     // AArch64 (big endian): aarch64_be
    aarch64_32,     // AArch64 (little endian) ILP32: aarch64_32
    arc,            // ARC: Synopsys ARC
    avr,            // AVR: Atmel AVR microcontroller
    bpfel,          // eBPF or extended BPF or 64-bit BPF (little endian)
    bpfeb,          // eBPF or extended BPF or 64-bit BPF (big endian)
    csky,           // CSKY: csky
    dxil,           // DXIL 32-bit DirectX bytecode
    hexagon,        // Hexagon: hexagon
    loongarch32,    // LoongArch (32-bit): loongarch32
    loongarch64,    // LoongArch (64-bit): loongarch64
    m68k,           // M68k: Motorola 680x0 family
    mips,           // MIPS: mips, mipsallegrex, mipsr6
    mipsel,         // MIPSEL: mipsel, mipsallegrexe, mipsr6el
    mips64,         // MIPS64: mips64, mips64r6, mipsn32, mipsn32r6
    mips64el,       // MIPS64EL: mips64el, mips64r6el, mipsn32el, mipsn32r6el
    msp430,         // MSP430: msp430
    ppc,            // PPC: powerpc
    ppcle,          // PPCLE: powerpc (little endian)
    ppc64,          // PPC64: powerpc64, ppu
    ppc64le,        // PPC64LE: powerpc64le
    r600,           // R600: AMD GPUs HD2XXX - HD6XXX
    amdgcn,         // AMDGCN: AMD GCN GPUs
    riscv32,        // RISC-V (32-bit): riscv32
    riscv64,        // RISC-V (64-bit): riscv64
    sparc,          // Sparc: sparc
    sparcv9,        // Sparcv9: Sparcv9
    sparcel,        // Sparc: (endianness = little). NB: 'Sparcle' is a CPU variant
    systemz,        // SystemZ: s390x
    tce,            // TCE (http://tce.cs.tut.fi/): tce
    tcele,          // TCE little endian (http://tce.cs.tut.fi/): tcele
    thumb,          // Thumb (little endian): thumb, thumbv.*
    thumbeb,        // Thumb (big endian): thumbeb
    x86,            // X86: i[3-9]86
    x86_64,         // X86-64: amd64, x86_64
    xcore,          // XCore: xcore
    nvptx,          // NVPTX: 32-bit
    nvptx64,        // NVPTX: 64-bit
    le32,           // le32: generic little-endian 32-bit CPU (PNaCl)
    le64,           // le64: generic little-endian 64-bit CPU (PNaCl)
    amdil,          // AMDIL
    amdil64,        // AMDIL with 64-bit pointers
    hsail,          // AMD HSAIL
    hsail64,        // AMD HSAIL with 64-bit pointers
    spir,           // SPIR: standard portable IR for OpenCL 32-bit version
    spir64,         // SPIR: standard portable IR for OpenCL 64-bit version
    spirv32,        // SPIR-V with 32-bit pointers
    spirv64,        // SPIR-V with 64-bit pointers
    kalimba,        // Kalimba: generic kalimba
    shave,          // SHAVE: Movidius vector VLIW processors
    lanai,          // Lanai: Lanai 32-bit
    wasm32,         // WebAssembly with 32-bit pointers
    wasm64,         // WebAssembly with 64-bit pointers
    renderscript32, // 32-bit RenderScript
    renderscript64, // 64-bit RenderScript
    ve,             // NEC SX-Aurora Vector Engine
    // LastArchType = ve
}

pub enum SubArchType {
    NoSubArch,

    ARMSubArch_v9_4a,
    ARMSubArch_v9_3a,
    ARMSubArch_v9_2a,
    ARMSubArch_v9_1a,
    ARMSubArch_v9,
    ARMSubArch_v8_9a,
    ARMSubArch_v8_8a,
    ARMSubArch_v8_7a,
    ARMSubArch_v8_6a,
    ARMSubArch_v8_5a,
    ARMSubArch_v8_4a,
    ARMSubArch_v8_3a,
    ARMSubArch_v8_2a,
    ARMSubArch_v8_1a,
    ARMSubArch_v8,
    ARMSubArch_v8r,
    ARMSubArch_v8m_baseline,
    ARMSubArch_v8m_mainline,
    ARMSubArch_v8_1m_mainline,
    ARMSubArch_v7,
    ARMSubArch_v7em,
    ARMSubArch_v7m,
    ARMSubArch_v7s,
    ARMSubArch_v7k,
    ARMSubArch_v7ve,
    ARMSubArch_v6,
    ARMSubArch_v6m,
    ARMSubArch_v6k,
    ARMSubArch_v6t2,
    ARMSubArch_v5,
    ARMSubArch_v5te,
    ARMSubArch_v4t,

    AArch64SubArch_arm64e,
    AArch64SubArch_arm64ec,

    KalimbaSubArch_v3,
    KalimbaSubArch_v4,
    KalimbaSubArch_v5,

    MipsSubArch_r6,

    PPCSubArch_spe,

    // SPIR-V sub-arch corresponds to its version.
    SPIRVSubArch_v10,
    SPIRVSubArch_v11,
    SPIRVSubArch_v12,
    SPIRVSubArch_v13,
    SPIRVSubArch_v14,
    SPIRVSubArch_v15,
}

pub enum VendorType {
    UnknownVendor,

    Apple,
    PC,
    SCEI,
    Freescale,
    IBM,
    ImaginationTechnologies,
    MipsTechnologies,
    NVIDIA,
    CSR,
    Myriad,
    AMD,
    Mesa,
    SUSE,
    OpenEmbedded,
    // LastVendorType = OpenEmbedded
}

pub enum OSType {
    UnknownOS,

    Ananas,
    CloudABI,
    Darwin,
    DragonFly,
    FreeBSD,
    Fuchsia,
    IOS,
    KFreeBSD,
    Linux,
    Lv2,        // PS3
    MacOSX,
    NetBSD,
    OpenBSD,
    Solaris,
    Win32,
    ZOS,
    Haiku,
    Minix,
    RTEMS,
    NaCl,       // Native Client
    AIX,
    CUDA,       // NVIDIA CUDA
    NVCL,       // NVIDIA OpenCL
    AMDHSA,     // AMD HSA Runtime
    PS4,
    PS5,
    ELFIAMCU,
    TvOS,       // Apple tvOS
    WatchOS,    // Apple watchOS
    DriverKit,  // Apple DriverKit
    Mesa3D,
    Contiki,
    AMDPAL,     // AMD PAL Runtime
    HermitCore, // HermitCore Unikernel/Multikernel
    Hurd,       // GNU/Hurd
    WASI,       // Experimental WebAssembly OS
    Emscripten,
    ShaderModel, // DirectX ShaderModel
    // LastOSType = ShaderModel
}

pub enum EnvironmentType {
    UnknownEnvironment,

    GNU,
    GNUABIN32,
    GNUABI64,
    GNUEABI,
    GNUEABIHF,
    GNUF32,
    GNUF64,
    GNUSF,
    GNUX32,
    GNUILP32,
    CODE16,
    EABI,
    EABIHF,
    Android,
    Musl,
    MuslEABI,
    MuslEABIHF,
    MuslX32,

    MSVC,
    Itanium,
    Cygnus,
    CoreCLR,
    Simulator, // Simulator variants of other systems, e.g., Apple's iOS
    MacABI, // Mac Catalyst variant of Apple's iOS deployment target.

    // Shader Stages
    // The order of these values matters, and must be kept in sync with the
    // language options enum in Clang. The ordering is enforced in
    // static_asserts in Triple.cpp and in Clang.
    Pixel,
    Vertex,
    Geometry,
    Hull,
    Domain,
    Compute,
    Library,
    RayGeneration,
    Intersection,
    AnyHit,
    ClosestHit,
    Miss,
    Callable,
    Mesh,
    Amplification,

    // LastEnvironmentType = Amplification
}

pub enum ObjectFormatType {
    UnknownObjectFormat,

    COFF,
    DXContainer,
    ELF,
    GOFF,
    MachO,
    SPIRV,
    Wasm,
    XCOFF,
}
