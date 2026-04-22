use std::fmt;
use std::ops::Not;

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
#[repr(transparent)]
pub struct Var(pub u32);

impl Var {
    #[inline]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

// Encoding: bit 0 = sign (1 = negated), upper bits = variable index.
// Negation is a single xor, and adjacent indices in watch/activity arrays
// are the positive and negative forms of one variable.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Lit(pub u32);

impl Lit {
    #[inline]
    pub fn new(var: Var, negated: bool) -> Self {
        Lit((var.0 << 1) | (negated as u32))
    }
    #[inline]
    pub fn var(self) -> Var {
        Var(self.0 >> 1)
    }
    #[inline]
    pub fn is_negated(self) -> bool {
        (self.0 & 1) == 1
    }
    #[inline]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
    #[inline]
    pub fn var_idx(self) -> usize {
        (self.0 >> 1) as usize
    }
}

impl Not for Lit {
    type Output = Lit;
    #[inline]
    fn not(self) -> Lit {
        Lit(self.0 ^ 1)
    }
}

impl fmt::Debug for Lit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_negated() {
            write!(f, "-{}", self.var().0 + 1)
        } else {
            write!(f, "{}", self.var().0 + 1)
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum LBool {
    True,
    False,
    Undef,
}

impl LBool {
    #[inline]
    pub fn from_bool(b: bool) -> Self {
        if b { LBool::True } else { LBool::False }
    }
    #[inline]
    pub fn negate(self) -> Self {
        match self {
            LBool::True => LBool::False,
            LBool::False => LBool::True,
            LBool::Undef => LBool::Undef,
        }
    }
}
