use std::collections::HashMap;
use crate::{bitwise::generate_mask, sif::{LinkType, LinkInfo}};


pub type LinkFn = fn(env: usize, mask: usize) -> f64;
// return probability that the future state of the element is ON

fn to_float(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}


fn link_copy(env: usize, mask: usize) -> f64 {
    to_float(env & mask > 0)
}

fn link_not(env: usize, mask: usize) -> f64 {
    to_float(env & mask == 0)
}

fn link_and(env: usize, mask: usize) -> f64 {
    to_float(env & mask == mask)
}

fn link_or(env: usize, mask: usize) -> f64 {
    to_float(env & mask > 0)
}

fn link_xor(env: usize, mask: usize) -> f64 {
    to_float((env & mask).count_ones() == 1)
}

fn link_any(env: usize, mask: usize) -> f64 {
    to_float(env & mask > 0)
}

fn link_all(env: usize, mask: usize) -> f64 {
    to_float(env & mask == mask)
}

fn link_even(env: usize, mask: usize) -> f64 {
    to_float((env & mask).count_ones() % 2 == 0)
}

fn link_odd(env: usize, mask: usize) -> f64 {
    to_float((env & mask).count_ones() % 2 == 1)
}

fn link_noisy(_env: usize, _mask: usize) -> f64 {
    0.5
}


pub fn get_link_fn(link: &LinkType, size: usize) -> LinkFn {
    match link {
        LinkType::COPY if size == 1 => link_copy,
        LinkType::NOT if size == 1 => link_not,
        LinkType::AND if size == 2 => link_and,
        LinkType::OR if size == 2 => link_or,
        LinkType::XOR if size == 2 => link_xor,
        LinkType::ANY if size > 0 => link_any,
        LinkType::ALL if size > 0 => link_all,
        LinkType::EVEN if size > 0 => link_even,
        LinkType::ODD if size > 0 => link_odd,
        LinkType::NOISY if size > 0 => link_noisy,

        _ => panic!("Not-implemented link type or invalid condition size"),
    }
}

pub fn get_link_fns(infos: Vec<LinkInfo>) -> Vec<(LinkFn, usize)> {
    let mut to_index = HashMap::<String, usize>::new();

    for (i, info) in infos.iter().enumerate() {
        let result = to_index.insert(info.element.clone(), i);

        if result.is_some() {
            panic!("Some element is defined twice or more: {:?}", info);
        }
    }


    let mut fns = Vec::<(LinkFn, usize)>::new();

    for info in infos.iter() {
        let mut indices = Vec::<usize>::new();

        for c in info.condition.iter() {
            let index = to_index.get(c);

            if let Some(x) = index {
                indices.push(*x);
            } else {
                panic!("ELEMENT '{}' has condition whose element is not defined", info.element);
            }
        }

        let link_fn = get_link_fn(&info.link_type, indices.len());
        let mask = generate_mask(&indices);

        fns.push((link_fn, mask));
    }

    fns
}
