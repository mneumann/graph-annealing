extern crate evo;
extern crate rand;
extern crate petgraph;
extern crate triadic_census;
extern crate graph_layout;
extern crate graph_neighbor_matching;
extern crate graph_edge_evolution;
extern crate graph_sgf;
extern crate serde_json;
extern crate sexp;

pub mod goal;
pub mod helper;
pub mod repr;
pub mod owned_weighted_choice;
pub mod stat;
pub mod fitness_function;

#[macro_export]
macro_rules! defops {
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
                        Err(format!("Invalid opcode: {}", s))
                    }
                }
            }
        }
    }
}
