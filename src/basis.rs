use crate::bitwise::{USIZE_BASIS, generate_vectors_from_mask};


pub struct CombinationIterator<'a> {
    initial: usize,
    current: usize,
    index: Vec<usize>,
    parent: &'a BitBasis,
}

impl <'a> Iterator for CombinationIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current & self.parent.unused_mask != 0 {
            return None
        }

        (0..self.parent.dim).for_each(|i| {
             self.index[i] = if self.current & USIZE_BASIS[i] == 0 {
                0
            } else {
                1
            };
        });

        self.current += 1;


        let mut union = self.initial;

        self.index.iter().zip(&self.parent.vectors).for_each(|(&i, &b)|{
            if i != 0 {
                union |= b;
            }
        });

        Some(union)
    }
}

#[derive(Debug, Clone)]
pub struct BitBasis {
    pub dim: usize,
    pub codim: usize,
    pub max_dim: usize,
    pub unused_mask: usize,
    pub vectors: Vec<usize>,
}

impl BitBasis {
    pub fn construct_from_vectors(vectors: &[usize], max_dim: usize) -> BitBasis {
        let basis_dim = vectors.len();
        let mut cloned_vectors = Vec::<usize>::with_capacity(basis_dim);
        vectors.iter().for_each(|&x| cloned_vectors.push(x));

        BitBasis {
            dim: basis_dim,
            codim: max_dim - basis_dim,
            max_dim: max_dim,
            unused_mask: usize::MAX << basis_dim,
            vectors: cloned_vectors,
        }
    }

    pub fn construct_from_mask(mask: usize, max_dim: usize) -> BitBasis {
        BitBasis::construct_from_vectors(&generate_vectors_from_mask(mask), max_dim)
    }

    pub fn generate_complement_basis(&self) -> BitBasis {
        let union = self.vectors.iter().fold(0, |acc, &vector| acc | vector );
        let mut complement = Vec::<usize>::with_capacity(self.codim);

        USIZE_BASIS[0..self.max_dim].iter().for_each(|&vector| {
            if union & vector == 0 {
                complement.push(vector);
            }
        });

        BitBasis {
            dim: self.codim,
            codim: self.dim,
            max_dim: self.max_dim,
            unused_mask: usize::MAX << self.codim,
            vectors: complement,
        }
    }

    pub fn to_mask(&self) -> usize {
        self.vectors.iter().fold(0, |acc, &x| acc | x)
    }

    pub fn image_size(&self) -> usize {
        1 << self.dim
    }

    pub fn codim_image_size(&self) -> usize {
        1 << self.codim
    }

    pub fn max_image_size(&self) -> usize {
        1 << self.max_dim
    }

    pub fn fixed_state(&self, state: usize) -> usize {
        self.vectors.iter().fold(0, |acc, &vector| {
            acc | (state & vector)
        })
    }

    pub fn span(&self, initial: usize) -> CombinationIterator {
        CombinationIterator {
            initial: initial,
            current: 0,
            index: vec![0; self.dim],
            parent: self,
        }
    }

    pub fn null_basis(max_dim: usize) -> BitBasis {
        BitBasis {
            dim: 0,
            codim: max_dim,
            max_dim: max_dim,
            unused_mask: usize::MAX,
            vectors: vec![],
        }
    }

    pub fn sub_basis(&self, index: &[usize]) -> BitBasis {
        let dim = index.len();

        let mut vectors = Vec::<usize>::with_capacity(dim);
        index.iter().for_each(|&i| {
            vectors.push(self.vectors[i]);
        });

        BitBasis {
            dim: dim,
            codim: self.max_dim - dim,
            max_dim: self.max_dim,
            unused_mask: usize::MAX << dim,
            vectors: vectors,
        }
    }
}
