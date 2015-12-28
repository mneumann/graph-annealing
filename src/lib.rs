extern crate evo;
extern crate rand;
extern crate petgraph;
extern crate triadic_census;
extern crate graph_layout;
extern crate graph_neighbor_matching;
extern crate graph_edge_evolution;
extern crate closed01;

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

#[macro_export]
macro_rules! defops {
    ($name:ident; $( $key:ident ),+) => {
        def_str_enum!{$name; $($key),+}

        impl $name {
            #[allow(dead_code)]
            pub fn all() -> Vec<$name> {
                vec![ $($name::$key),+ ]
            }
            // XXX: move
            #[allow(dead_code)]
            pub fn uniform_distribution() -> Vec<::rand::distributions::Weighted<$name>> {
                Self::all().iter().map(|&item|
                    ::rand::distributions::Weighted{weight:1,item:item}).collect()
            }
        }

    }
}

pub mod goal;
pub mod helper;
pub mod repr;
pub mod owned_weighted_choice;
pub mod stat;
