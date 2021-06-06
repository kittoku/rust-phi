use crate::bitwise::USIZE_BASES;


#[derive(Debug)]
pub struct MechanismPartition {
    pub left_purview: Vec<usize>,
    pub left_mechanism: Vec<usize>,
    pub right_purview: Vec<usize>,
    pub right_mechanism: Vec<usize>,
}
pub struct MechanismPartitionIterator {
    current: usize,
    purview_size: usize,
    mechanism_size: usize,
    mask_size: usize,
    unused_mask: usize,
}

impl MechanismPartitionIterator {
    pub fn construct(purview_size: usize, mechanism_size: usize) -> MechanismPartitionIterator {
        let mask_size = purview_size + mechanism_size;

        let unused_mask = usize::MAX << (mask_size - 1);

        MechanismPartitionIterator {
            current: 1, // 0 means no partition
            purview_size: purview_size,
            mechanism_size: mechanism_size,
            mask_size: mask_size,
            unused_mask: unused_mask,
        }
    }
}

impl Iterator for MechanismPartitionIterator {
    type Item = MechanismPartition;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current & self.unused_mask != 0 {
            return None
        }

        let mut left_purview = Vec::<usize>::new();
        let mut left_mechanism = Vec::<usize>::new();
        let mut right_purview = Vec::<usize>::new();
        let mut right_mechanism = Vec::<usize>::new();

        (0..self.mechanism_size).for_each(|i| {
            if self.current & USIZE_BASES[i] == 0 {
                left_mechanism.push(i);
            } else {
                right_mechanism.push(i);
            }
        });

        (0..self.purview_size).zip(self.mechanism_size..self.mask_size).for_each(|(i, j)| {
            if self.current & USIZE_BASES[j] == 0 {
                left_purview.push(i);
            } else {
                right_purview.push(i);
            }
        });

        self.current += 1;

        Some(MechanismPartition {
            left_purview: left_purview,
            left_mechanism: left_mechanism,
            right_purview: right_purview,
            right_mechanism: right_mechanism,
        })
    }
}