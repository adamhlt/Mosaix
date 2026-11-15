//! Serpentine guidance: which way to move next, row ends and the end of the map.
//!
//! The PCB size is unknown, so the guide only follows what the user confirms:
//! `Row` (move along the row) → end of row → `Closing` (capture the last tile of the row)
//! → `Down` (step to the next row) → first capture of the new row → `Row` in the opposite
//! direction. `finish` ends the map the same way (closing capture, then nothing more to do).

/// What happens once the closing capture is done (or skipped).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Then {
    StepDown,
    Finish,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Moving along the current row.
    Row,
    /// Row (or map) end confirmed: hold still so the last tile can be captured.
    Closing(Then),
    /// Stepping down to the next row.
    Down,
}

/// What the automatic detector proposes, depending on the phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Suggestion {
    /// Edge reached while moving along a row: end the row?
    EndRow,
    /// Edge reached while stepping down: finish the map?
    Finish,
}

#[derive(Debug, Clone)]
pub struct Guide {
    /// +1 = rows run to the right, -1 = to the left.
    pub row_dir: f64,
    pub phase: Phase,
    /// Rows completed so far.
    pub rows_done: usize,
    pub finished: bool,
}

impl Default for Guide {
    fn default() -> Self {
        Self { row_dir: 1.0, phase: Phase::Row, rows_done: 0, finished: false }
    }
}

impl Guide {
    /// Direction the camera should move (unit axis vector), `None` once finished.
    pub fn heading(&self) -> Option<(f64, f64)> {
        if self.finished {
            return None;
        }
        Some(match self.phase {
            Phase::Row | Phase::Closing(_) => (self.row_dir, 0.0),
            Phase::Down => (0.0, 1.0),
        })
    }

    /// Which suggestion the detector may raise right now.
    pub fn suggestion_kind(&self) -> Option<Suggestion> {
        if self.finished {
            return None;
        }
        match self.phase {
            Phase::Row => Some(Suggestion::EndRow),
            Phase::Down => Some(Suggestion::Finish),
            Phase::Closing(_) => None,
        }
    }

    /// The user confirmed the end of the row. `needs_closing`: whether a last tile must still be
    /// captured (there is new area since the last tile).
    pub fn end_row(&mut self, needs_closing: bool) {
        if self.finished || self.phase != Phase::Row {
            return;
        }
        if needs_closing {
            self.phase = Phase::Closing(Then::StepDown);
        } else {
            self.phase = Phase::Down;
            self.rows_done += 1;
        }
    }

    /// The user finished the map (from a row, or from the step down).
    pub fn finish(&mut self, needs_closing: bool) {
        if self.finished {
            return;
        }
        if needs_closing && !matches!(self.phase, Phase::Closing(Then::Finish)) {
            self.phase = Phase::Closing(Then::Finish);
        } else if !needs_closing {
            self.finished = true;
        }
    }

    /// Manual override: flip the direction of the rows.
    pub fn reverse(&mut self) {
        self.row_dir = -self.row_dir;
    }

    /// Resume mapping after a finish.
    pub fn resume(&mut self) {
        self.finished = false;
        self.phase = Phase::Row;
    }

    /// The closing capture was taken (or skipped).
    pub fn closing_done(&mut self) {
        if let Phase::Closing(then) = self.phase {
            match then {
                Then::StepDown => self.phase = Phase::Down,
                Then::Finish => {
                    self.phase = Phase::Row;
                    self.finished = true;
                }
            }
            if then == Then::StepDown {
                self.rows_done += 1;
            }
        }
    }

    /// A tile was captured outside of a closing: the first tile of the new row flips the direction.
    pub fn tile_captured(&mut self) {
        match self.phase {
            Phase::Closing(_) => self.closing_done(),
            Phase::Down => {
                self.phase = Phase::Row;
                self.row_dir = -self.row_dir;
            }
            Phase::Row => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serpentine_cycle() {
        let mut g = Guide::default();
        assert_eq!(g.heading(), Some((1.0, 0.0)));
        g.end_row(true);
        assert_eq!(g.phase, Phase::Closing(Then::StepDown));
        assert_eq!(g.heading(), Some((1.0, 0.0))); // still holding the row direction
        g.tile_captured(); // the closing capture
        assert_eq!(g.phase, Phase::Down);
        assert_eq!(g.rows_done, 1);
        assert_eq!(g.heading(), Some((0.0, 1.0)));
        g.tile_captured(); // first tile of the next row
        assert_eq!(g.phase, Phase::Row);
        assert_eq!(g.heading(), Some((-1.0, 0.0)));
    }

    #[test]
    fn end_row_without_new_area_skips_closing() {
        let mut g = Guide::default();
        g.end_row(false);
        assert_eq!(g.phase, Phase::Down);
    }

    #[test]
    fn finish_with_and_without_closing_capture() {
        let mut g = Guide::default();
        g.finish(true);
        assert_eq!(g.phase, Phase::Closing(Then::Finish));
        assert!(!g.finished);
        g.tile_captured();
        assert!(g.finished);
        assert_eq!(g.heading(), None);
        assert_eq!(g.suggestion_kind(), None);

        let mut g = Guide::default();
        g.finish(false);
        assert!(g.finished);
        g.resume();
        assert!(!g.finished);
        assert_eq!(g.heading(), Some((1.0, 0.0)));
    }

    #[test]
    fn suggestion_kind_follows_the_phase() {
        let mut g = Guide::default();
        assert_eq!(g.suggestion_kind(), Some(Suggestion::EndRow));
        g.end_row(false);
        assert_eq!(g.suggestion_kind(), Some(Suggestion::Finish));
        g.end_row(true); // ignored outside of a row
        assert_eq!(g.phase, Phase::Down);
    }

    #[test]
    fn reverse_flips_row_direction() {
        let mut g = Guide::default();
        g.reverse();
        assert_eq!(g.heading(), Some((-1.0, 0.0)));
    }
}
