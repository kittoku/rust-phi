use std::{mem::size_of, usize};


pub const USIZE_BITS: usize = size_of::<usize>() * 8;
pub const USIZE_BASIS: [usize; 64] = [
    1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7, 1 << 8, 1 << 9,
    1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19,
    1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24, 1 << 25, 1 << 26, 1 << 27, 1 << 28, 1 << 29,
    1 << 30, 1 << 31, 1 << 32, 1 << 33, 1 << 34, 1 << 35, 1 << 36, 1 << 37, 1 << 38, 1 << 39,
    1 << 40, 1 << 41, 1 << 42, 1 << 43, 1 << 44, 1 << 45, 1 << 46, 1 << 47, 1 << 48, 1 << 49,
    1 << 50, 1 << 51, 1 << 52, 1 << 53, 1 << 54, 1 << 55, 1 << 56, 1 << 57, 1 << 58, 1 << 59,
    1 << 60, 1 << 61, 1 << 62, 1 << 63];

pub fn generate_mask(indices: &Vec::<usize>) -> usize {
    let mut mask: usize = 0;

    for index in indices {
        mask |= 1 << index;
    }

    mask
}

pub fn generate_indices(mask: usize) -> Vec<usize> {
    let mut indices = Vec::<usize>::new();
    let mut digit = 1;

    (0..USIZE_BITS).for_each(|i| {
        if mask & digit != 0 {
            indices.push(i);
        }

        digit <<= 1;
    });

    indices
}

pub fn generate_vectors_from_indices(indices: &Vec<usize>) -> Vec<usize> {
    indices.iter().map(|i| USIZE_BASIS[*i]).collect()
}

pub fn generate_vectors_from_mask(mask: usize) -> Vec<usize> {
    let indices = generate_indices(mask);
    generate_vectors_from_indices(&indices)
}
