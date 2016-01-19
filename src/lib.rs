extern crate evo;
extern crate rand;
extern crate petgraph;
extern crate triadic_census;
extern crate graph_layout;
extern crate graph_neighbor_matching;
extern crate graph_edge_evolution;
extern crate closed01;
extern crate graph_io_gml;
extern crate asexp;

/// Defines a public enum that can be converted to and from a string.
#[macro_export]
macro_rules! def_str_enum {
    ($name:ident; $( $key:ident ),+) => {
        #[derive(Copy, Clone, PartialEq, Eq, Debug)]
        pub enum $name {
            $(
                $key
             ),+
        }

        impl ::std::string::ToString for $name {
            fn to_string(&self) -> String {
                match *self {
                    $(
                        $name::$key => stringify!($key).to_string()
                    ),+
                }
            }
        }

        impl ::std::str::FromStr for $name {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $(
                        stringify!($key) => {
                            Ok($name::$key)
                        }
                     )+
                    _ => {
                        Err(format!("Invalid string: {}", s))
                    }
                }
            }
        }
    }
}

pub trait UniformDistribution: Sized {
    fn uniform_distribution() -> Vec<(Self, u32)>;
}

#[macro_export]
macro_rules! defops {
    ($name:ident; $( $key:ident ),+) => {
        def_str_enum!{$name; $($key),+}

        impl $name {
            #[allow(dead_code)]
            pub fn all() -> Vec<$name> {
                vec![ $($name::$key),+ ]
            }
        }

        impl ::UniformDistribution for $name {
            fn uniform_distribution() -> Vec<($name, u32)> {
                Self::all().iter().map(|&item| (item, 1)).collect()
            }
        }

    }
}

pub mod goal;
pub mod helper;
pub mod repr;
pub mod owned_weighted_choice;
pub mod stat;
pub mod graph;
pub mod update;
