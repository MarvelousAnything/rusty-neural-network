use std::ops::{Add, Div, Neg};
use ndarray::{arr1, Array1, Array2};
use ndarray_rand::RandomExt;
use rand::{Rng, thread_rng};
use rand::distributions::Uniform;

fn sigmoid(x: f64) -> f64 {
    x.neg().exp().add(1.0).powi(-1)
}

fn sigmoid_derivative(x: f64) -> f64 {
    x.exp().div(x.exp().add(1.0).powi(2))
}

#[derive(Debug)]
struct Layer {
    size: usize,
    bias: f64,
    weights: Array2<f64>
}

impl Layer {
    pub fn new(size: usize, prev_size: usize) -> Layer {
        let bias: f64 = thread_rng().gen_range(0.0..1.0);
        let weights: Array2<f64> = Array2::random((prev_size, size), Uniform::new(0.0, 1.0));
        Layer { size, bias, weights }
    }
}

#[derive(Debug)]
struct Network {
    topology: Vec<usize>,
    topology_size: usize,
    input_size: usize,
    output_size: usize,
    layers: Vec<Layer>
}

impl Network {
    pub fn new(topology: Vec<usize>) -> Network {
        let mut layers: Vec<Layer> = Vec::new();
        let topology_size = topology.len();
        for size in 0..topology_size-1 {
            layers.push(Layer::new(topology[size + 1], topology[size]));
        }
        let input_size: usize = topology[0];
        let output_size: usize = topology[topology.len() - 1];
        Network { topology, topology_size, input_size, output_size, layers }
    }

    pub fn think(&self, input_data: Array1<f64>) -> Vec<Array1<f64>> {
        let mut input: Array1<f64> = input_data;
        let mut outputs: Vec<Array1<f64>> = Vec::new();
        let mut counter = 0;
        println!("{counter}\t{input:?}");
        for layer in self.layers.iter() {
            input = layer.weights.t().dot(&input) + layer.bias;
            input = input.map(|x| sigmoid(*x));
            outputs.push(input.clone());
            counter += 1;
            println!("{counter}\t{input:?}");
        }
        outputs
    }

    pub fn train(&self, training_inputs: Array1<f64>, training_outputs: Array1<f64>, epochs: usize) {
        for _ in 0..epochs {
            let outputs = self.think(training_inputs.clone());
            let mut output_error: Array1<f64> = &training_outputs - &outputs[self.topology_size-2];

            for i in (0..self.topology_size-1).rev() {
                let out = &outputs[i];
                let weights = &self.layers[i].weights;
                let gradients = out.map(|x| sigmoid_derivative(*x)) * &output_error;

                println!("\n{:-^20}", self.topology_size-3 + i);
                println!("out: {:#?}", out);
                println!("weights: {:#?}", weights);
                println!("\ngradients: {gradients:#?}");
                println!("\nerror: {output_error:#?}");
                println!("\ntest: {:#?}", gradients.dot(&weights.t()));

                output_error = outputs[i - 1].clone();
            }

            // training_inputs.clone().dot(&(output.map(|x| sigmoid_derivative(*x)) * error));
        }
    }
}

fn main() {
    let network = Network::new(vec![2, 4, 8, 4]);
    let inputs: Array1<f64> = arr1(&[0.3, 0.2]);
    let outputs: Array1<f64> = arr1(&[0.9, 0.3, 0.5, 0.6]);
    // println!("{:#?}", network);
    network.train(inputs, outputs, 1);
    // println!("{:#?}", network.think(inputs));
}