//! This file contains an analysis for computing the multi-level callgraph from a given top-level operation. This nodes within this callgraph are defined by the `CallOpInterface` and `CallableOpInterface` operation interfaces defined in CallInterface.td.
//! 
//! This file contains interfaces and analyses for defining a nested callgraph.
//! 
//! - include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Analysis/CallGraph.h>
//! - lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/Analysis/CallGraph.cpp>

use core::{
    cmp::PartialEq,
    iter::IntoIterator
};
use llvm::{
    adt::{
        dense_set::SmallDenseSet,
        graph_traits,
        map_vector::MapVector,
        pointer_int_pair::PointerIntPair,
        pointer_union,
        scc_iterator,
        set_vector::SetVector,
        small_vector::SmallVector
    },
    support::raw_ostream
};
use crate::{
    interfaces::call_interfaces,
    ir::{
        operation,
        region::Region,
        symbol_table
    },
    support::llvm
};

/// This class represents a single callable in the callgraph. Aside from the external node, each node represents a callable node in the graph and contains a valid corresponding Region. The external node is a virtual node used to represent external edges into, and out of, the callgraph.
pub struct CallGraphNode {
    /// The callable region defines the boundary of the call graph node. This is the region referenced by 'call' operations. This is at a per-region boundary as operations may define multiple callable regions.
    callable_region: Option<Region>,

    /// A set of out-going edges from this node to other nodes in the graph.
    edges: SetVector<Edge, SmallVector<[Edge; 4]>,
            SmallDenseSet<Edge, 4, EdgeKeyInfo>>
}

impl CallGraphNode {
    /// Returns true if this node is an external node.
    pub const fn is_external(&self) -> bool {
        self.callable_region.is_none()
    }

    /// Returns the callable region this node represents. This can only be called
    /// on non-external nodes.
    pub const fn callable_region(&self) -> Option<Region> {
        assert!(!self.is_external(),
                "The external node has no callable region.");
        self.callable_region
    }

    // /**
    // Adds an abstract reference edge to the given node. An abstract edge does
    // not come from any observable operations, so this is only valid on the
    // external node.
    // */
    // pub fn add_abstract_edge(&mut self, node: Option<CallGraphNode>) {
    //     assert!(is_external(), "Abstract edges are only valid on external nodes.");
    //     self.add_edge(node, EdgeKind::Abstract);
    // }

    // /// Add an outgoing call edge from this node.
    // pub fn add_call_edge(&mut self, node: Option<CallGraphNode>) {
    //     self.add_edge(node, EdgeKind::Call)
    // }

    // /// Adds a reference edge to the given child node.
    // pub fn add_child_edge(&mut self, child: Option<CallGraphNode>) {
    //     self.add_edge(child, EdgeKind::Child);
    // }

    // /// Returns true if this node has any child edges.
    // pub const fn has_children(&self) -> bool {
    //     any_of(edges, |edge: &Edge| edge.is_child())
    // }

    // /// Add an edge to 'node' with the given kind.
    // fn add_edge(&mut self, node: Option<CallGraphNode>, kind: EdgeKind) {
    //     self.edges.insert(node, kind);
    // }
}

// // const_iterator
// /// Iterator over the outgoing edges of this node.
// impl IntoIterator for CallGraphNode {
//     type Item = Edge;
//     type IntoIter = SmallVector<Edge>;

//     fn into_iter(self) -> Self::IntoIter {
//         SmallVector::new(edges).iter()  // TODO
//     }
// }


/// This class represents a directed edge between two nodes in the callgraph.
pub struct Edge {
    /// The target node of this edge, as well as the edge kind.
    target_and_kind: PointerIntPair<Option<CallGraphNode>, 2, EdgeKind>
}

impl Edge {
    // /// Returns true if this edge represents an `Abstract` edge.
    // pub const fn is_abstract(&self) -> bool {
    //     self.target_and_kind.get_int() == EdgeKind::Abstract
    // }

    // /// Returns true if this edge represents a `Call` edge.
    // pub const fn is_call(&self) -> bool {
    //     self.target_and_kind.get_int() == EdgeKind::Call
    // }

    // /// Returns true if this edge represents a `Child` edge.
    // pub const fn is_child(&self) -> bool {
    //     self.target_and_kind.get_int() == EdgeKind::Child
    // }

    // /// Returns the target node for this edge.
    // pub const fn get_target(&self) -> Option<CallGraphNode> {
    //     self.target_and_kind.get_pointer()
    // }
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.target_and_kind == other.target_and_kind
    }
}

enum EdgeKind {
    /// An 'Abstract' edge represents an opaque, non-operation, reference between this node and the target. Edges of this type are only valid from the external node, as there is no valid connection to an operation in the module.
    Abstract,

    /// A 'Call' edge represents a direct reference to the target node via a
    /// call-like operation within the callable region of this node.
    Call,

    /// A 'Child' edge is used when the region of target node is defined inside of the callable region of this node. This means that the region of this node is an ancestor of the region for the target node. As such, this edge cannot be used on the 'external' node.
    Child,
}

/// DenseMap info for callgraph edges.
struct EdgeKeyInfo {
}

pub struct CallGraph {
    /// The set of nodes within the callgraph.
    nodes: NodeMap,

    /// A special node used to indicate an external caller.
    external_caller_node: CallGraphNode,

    /// A special node used to indicate an unknown callee.
    unknown_callee_node: CallGraphNode
}

type NodeMap = MapVector<Option<Region>, Box<CallGraphNode>>;
