#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }

    fn can_hold(&self, other: &Rectangle) -> bool {
        self.height > other.height && self.width > other.width
    }

    fn square(size: u32) -> Rectangle {
        Rectangle { width: size, height: size }
    }
}

fn main() {
    let rect1 = Rectangle { width: 30, height: 50 };
    let rect2 = Rectangle::square(30);

    println!("Can rect1 hold rect2: {}", &rect1.can_hold(&rect2));
    println!("Can rect2 hold rect1: {}", &rect2.can_hold(&rect1));

    println!(
        "The area of the rectangle is {} square pixels",
        rect1.area()
    );
    
}
