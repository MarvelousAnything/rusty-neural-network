mod layer;
mod network;
mod cost;
mod activation;
mod neuron;
mod trainer;
mod model;

use crate::layer::Layer;
use crate::network::Network;

fn main() {
    let network = Network::new(&[2, 3, 2]);
    println!("{:?}", network);
    let outputs = network.calculate_outputs(&[1.0, 2.0]).unwrap();
    println!("{:?}", outputs);
    let cost = network.calculate_cost(&[1.0, 2.0], &[1.0, 0.0]).unwrap();
    println!("{:?}", cost);
}
