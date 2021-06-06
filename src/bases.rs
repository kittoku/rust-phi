use crate::bitwise::{USIZE_BASES, generate_bases_from_mask};


pub struct CombinationIterator<'a> {
    initial: usize,
    current: usize,
    index: Vec<usize>,
    parent: &'a BitBases,
}

impl <'a> Iterator for CombinationIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current & self.parent.unused_mask != 0 {
            return None
        }

        (0..self.parent.dim).for_each(|i| {
             self.index[i] = if self.current & USIZE_BASES[i] == 0 {
                0
            } else {
                1
            };
        });

        self.current += 1;


        let mut union = self.initial;

        self.index.iter().zip(&self.parent.bases).for_each(|(&i, &b)|{
            if i != 0 {
                union |= b;
            }
        });

        Some(union)
    }
}

#[derive(Debug, Clone)]
pub struct BitBases {
    pub dim: usize,
    pub codim: usize,
    pub max_dim: usize,
    pub unused_mask: usize,
    pub bases: Vec<usize>,
}

impl BitBases {
    pub fn construct_from_bases(bases: &[usize], max_dim: usize) -> BitBases {
        let bases_dim = bases.len();
        let mut cloned_bases = Vec::<usize>::with_capacity(bases_dim);
        bases.iter().for_each(|&x| cloned_bases.push(x));

        BitBases {
            dim: bases_dim,
            codim: max_dim - bases_dim,
            max_dim: max_dim,
            unused_mask: usize::MAX << bases_dim,
            bases: cloned_bases,
        }
    }

    pub fn construct_from_mask(mask: usize, max_dim: usize) -> BitBases {
        BitBases::construct_from_bases(&generate_bases_from_mask(mask), max_dim)
    }

    pub fn generate_complement_bases(&self) -> BitBases {
        let union = self.bases.iter().fold(0, |acc, &base| acc | base );
        let mut complement = Vec::<usize>::with_capacity(self.codim);

        USIZE_BASES[0..self.max_dim].iter().for_each(|&base| {
            if union & base == 0 {
                complement.push(base);
            }
        });

        BitBases {
            dim: self.codim,
            codim: self.dim,
            max_dim: self.max_dim,
            unused_mask: usize::MAX << self.codim,
            bases: complement,
        }
    }

    pub fn to_mask(&self) -> usize {
        self.bases.iter().fold(0, |acc, &x| acc | x)
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
        self.bases.iter().fold(0, |acc, &base| {
            acc | (state & base)
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

    pub fn null_bases(max_dim: usize) -> BitBases {
        BitBases {
            dim: 0,
            codim: max_dim,
            max_dim: max_dim,
            unused_mask: usize::MAX,
            bases: vec![],
        }
    }

    pub fn sub_bases(&self, index: &[usize]) -> BitBases {
        let dim = index.len();

        let mut bases = Vec::<usize>::with_capacity(dim);
        index.iter().for_each(|&i| {
            bases.push(self.bases[i]);
        });

        BitBases {
            dim: dim,
            codim: self.max_dim - dim,
            max_dim: self.max_dim,
            unused_mask: usize::MAX << dim,
            bases: bases,
        }
    }
}
