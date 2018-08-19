use std::env;
use std::collections::LinkedList;

fn main() {
    let mut args: LinkedList<String> = env::args().collect();
    args.pop_front();

    while let Some(s) = args.pop_front() {
        print!("{} ", s);
    }

    println!("");
}