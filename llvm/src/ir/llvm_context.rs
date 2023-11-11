/*!
# Class for Managing 'Global' State

This file declares LLVMContext, a container of "global" state in LLVM, such as the global type and constant uniquing tables.

This file implements LLVMContext, as a wrapper around the opaque class LLVMContextImpl.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/LLVMContext.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/IR/LLVMContext.cpp>
*/

use crate::llvm::{
    adt::{
        dense_map::DenseMap,
        small_ptr_set::SmallPtrSet,
        string_map::StringMap
    },
    ir::module::Module
};

/**
This is an important class for using LLVM in a threaded context.  It (opaquely) owns and manages the core "global" data of LLVM's core infrastructure, including the type and constant uniquing tables.
LLVMContext itself provides no locking guarantees, so you should be careful to have one context per thread.
*/
pub struct LLVMContext {
    // ===============
    // LLVMContextImpl
    // ===============

    // /**
    // OwnedModules - The set of modules instantiated in this context, and which will be automatically deleted if this context is deleted.
    // */
    // owned_modules: SmallPtrSet<Module *, 4>,

    /**
    The main remark streamer used by all the other streamers (e.g. IR, MIR, frontends, etc.). This should only be used by the specific streamers, and never directly.
    */
    // main_remark_streamer: Box<remarks::RemarkStreamer>,

    // diag_handler: Box<DiagnosticHandler>,
    respect_diagnostic_filters: bool,  // = false;
    diagnostics_hotness_requested: bool,  // = false;

    /**
    The minimum hotness value a diagnostic needs in order to be included in optimisation diagnostics.

    The threshold is an Optional value, which maps to one of the 3 states:
    1). 0            => threshold disabled. All emarks will be printed.
    2). positive int => manual threshold by user. Remarks with hotness exceed
                        threshold will be printed.
    3). None         => 'auto' threshold by user. The actual value is not
                        available at command line, but will be synced with
                        hotness threhold from profile summary during
                        compilation.

    State 1 and 2 are considered as terminal states. State transition is only allowed from 3 to 2, when the threshold is first synced with profile summary. This ensures that the threshold is set only once and stays constant.

    If threshold option is not specified, it is disabled (0) by default.
    */
    diagnostics_hotness_threshold: Option<u64>,  // = 0;

    /// The percentage of difference between profiling branch weights and
    /// llvm.expect branch weights to tolerate when emiting MisExpect diagnostics
    diagnostics_mis_expect_tolerance: Option<u32>,  // = 0;
    mis_expect_warning_requested: bool,  // = false;

    // /// The specialised remark streamer used by LLVM's OptimizationRemarkEmitter.
    // llvmrs: Box<LLVMRemarkStreamer>,

    // yield_callback: LLVMContext::YieldCallbackTy,  // = nullptr;
    // yield_opaque_handle: void *,  // = nullptr;

    // value_names: DenseMap<const Value *, ValueName *>,

    // using IntMapTy =
    //     DenseMap<APInt, Box<ConstantInt>, DenseMapAPIntKeyInfo>;
    // IntMapTy IntConstants;

    // using FPMapTy =
    //     DenseMap<APFloat, Box<ConstantFP>, DenseMapAPFloatKeyInfo>;
    // FPMapTy FPConstants;

    // attrs_set: FoldingSet<AttributeImpl>,
    // attrs_lists: FoldingSet<AttributeListImpl>,
    // attrs_set_nodes: FoldingSet<AttributeSetNode>,

    // md_string_cache: StringMap<MDString, BumpPtrAllocator>,
    // values_as_metadata: DenseMap<Value *, ValueAsMetadata *>,
    // metadata_as_values: DenseMap<Metadata *, MetadataAsValue *>,

    // #define HANDLE_MDNODE_LEAF_UNIQUABLE(CLASS)                                    \
    // DenseSet<CLASS *, CLASS##Info> CLASS##s;
    // #include "llvm/IR/Metadata.def"

    // // Optional map for looking up composite types by identifier.
    // di_type_map: Option<DenseMap<const MDString *, DICompositeType *>>,

    // /**
    // MDNodes may be uniqued or not uniqued.  When they're not uniqued, they aren't in the MDNodeSet, but they're still shared between objects, so no one object can destroy them.  Keep track of them here so we can delete them on context teardown.
    // */
    // distinct_md_nodes: Vec<MDNode *>,

    // caz_constants: DenseMap<Type *, Box<ConstantAggregateZero>>,

    // using ArrayConstantsTy = ConstantUniqueMap<ConstantArray>;
    // ArrayConstantsTy ArrayConstants;

    // using StructConstantsTy = ConstantUniqueMap<ConstantStruct>;
    // StructConstantsTy StructConstants;

    // using VectorConstantsTy = ConstantUniqueMap<ConstantVector>;
    // VectorConstantsTy VectorConstants;

    // cpn_constants: DenseMap<PointerType *, Box<ConstantPointerNull>>,

    // ctn_constants: DenseMap<TargetExtType *, Box<ConstantTargetNone>>,

    // uv_constants: DenseMap<Type *, Box<UndefValue>>,

    // pv_constants: DenseMap<Type *, Box<PoisonValue>>,

    // cds_constants: StringMap<Box<ConstantDataSequential>>,

    // block_addresses: DenseMap<(const Function *, const BasicBlock *), BlockAddress *>,

    // dso_local_equivalents: DenseMap<const GlobalValue *, DSOLocalEquivalent *>,

    // no_cfi_values: sDenseMap<const GlobalValue *, NoCFIValue *>,

    // expr_constants; ConstantUniqueMap<ConstantExpr>,

    // inline_asms: ConstantUniqueMap<InlineAsm>,

    // the_true_val: Option<ConstantInt>,  // = nullptr;
    // the_false_val: Option<ConstantInt>,  // = nullptr;

    // // Basic type instances.
    // Type VoidTy, LabelTy, HalfTy, BFloatTy, FloatTy, DoubleTy, MetadataTy,
    //     TokenTy;
    // Type X86_FP80Ty, FP128Ty, PPC_FP128Ty, X86_MMXTy, X86_AMXTy;
    // IntegerType Int1Ty, Int8Ty, Int16Ty, Int32Ty, Int64Ty, Int128Ty;

    // the_none_token: Box<ConstantTokenNone>,

    // alloc: BumpPtrAllocator,
    // saver: UniqueStringSaver,  //{Alloc};

    // integer_types: DenseMap<unsigned, IntegerType *>,

    // using FunctionTypeSet = DenseSet<FunctionType *, FunctionTypeKeyInfo>;
    // FunctionTypeSet FunctionTypes;
    // using StructTypeSet = DenseSet<StructType *, AnonStructTypeKeyInfo>;
    // StructTypeSet AnonStructTypes;
    // named_struct_types: StringMap<StructType *>,
    // named_struct_types_unique_id: unsigned,  // = 0;

    // using TargetExtTypeSet = DenseSet<TargetExtType *, TargetExtTypeKeyInfo>;
    // TargetExtTypeSet TargetExtTypes;

    // DenseMap<std::pair<Type *, uint64_t>, ArrayType *> ArrayTypes;
    // DenseMap<std::pair<Type *, ElementCount>, VectorType *> VectorTypes;
    // DenseMap<Type *, PointerType *> PointerTypes; // Pointers in AddrSpace = 0
    // DenseMap<std::pair<Type *, unsigned>, PointerType *> ASPointerTypes;
    // DenseMap<std::pair<Type *, unsigned>, TypedPointerType *> ASTypedPointerTypes;

    // /// ValueHandles - This map keeps track of all of the value handles that are
    // /// watching a Value*.  The Value::HasValueHandle bit is used to know
    // /// whether or not a value has an entry in this map.
    // using ValueHandlesTy = DenseMap<Value *, ValueHandleBase *>;
    // ValueHandlesTy ValueHandles;

    // /// CustomMDKindNames - Map to hold the metadata string to ID mapping.
    // custom_md_kind_names: StringMap<unsigned>,

    // /// Collection of metadata used in this context.
    // value_metadata: DenseMap<const Value *, MDAttachments>,

    // /**
    // Map DIAssignID -> Instructions with that attachment.
    // Managed by Instruction via Instruction::updateDIAssignIDMapping.
    // Query using the at:: functions defined in DebugInfo.h.
    // */
    // assignment_id_to_instrs: DenseMap<DIAssignID *, SmallVector<Instruction *, 1>>,

    // /// Collection of per-GlobalObject sections used in this context.
    // global_object_sections: DenseMap<const GlobalObject *, &'static str>,

    // /// Collection of per-GlobalValue partitions used in this context.
    // global_value_partitions: DenseMap<const GlobalValue *, &'static str>,

    // global_value_sanitiser_metadata: DenseMap<const GlobalValue *, GlobalValue::SanitiserMetadata>,

    // /**
    // DiscriminatorTable - This table maps file:line locations to an integer representing the next DWARF path discriminator to assign to instructions in different blocks at the same location.
    // */
    // discriminator_table: DenseMap<(const char *, unsigned), unsigned>,

    /**
    A set of interned tags for operand bundles.  The StringMap maps bundle tags to their IDs.

    See LLVMContext::getOperandBundleTagID
    */
    bundle_tag_cache: StringMap<u32>,

    // fn get_or_insert_bundle_tag(tag: &str) -> Option<StringMapEntry<u32>>;
    // const fn get_operand_bundle_tags(tags: &SmallVectorImpl<&str>);
    // const fn get_operand_bundle_tag_id(tag: &str) -> u32;

    // /**
    // A set of interned synchronization scopes. The StringMap maps synchronisation scope names to their respective synchronization scope IDs.
    // */
    // ssc: StringMap<SyncScope::ID>,

    // /**
    // getOrInsertSyncScopeID - Maps synchronization scope name to synchronisation scope ID. Every synchronisation scope registered with LLVMContext has unique ID except pre-defined ones.
    // */
    // SyncScope::ID get_or_insert_sync_scope_id(SSN: &str);

    // /**
    // getSyncScopeNames - Populates client supplied SmallVector with synchronisation scope names registered with LLVMContext. Synchronisation scope names are ordered by increasing synchronization scope IDs.
    // */
    // cons fn sync_scope_names(SSNs: &SmallVector<&str>);

    /**
    Maintain the GC name for each function.

    This saves allocating an additional word in Function for programs which do not use GC (i.e., most programs) at the cost of increased overhead for clients which do use GC.
    */
    // gc_names: DenseMap<const Function *, String>,

    /**
    Flag to indicate if Value (other than GlobalValue) retains their name or not.
    */
    discard_value_names: bool  // = false;
}

impl LLVMContext {
    
}
