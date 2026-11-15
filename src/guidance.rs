//! Serpentine guidance: which way to move next, row ends and the end of the map.
//!
//! The user says when a row ends and when the map is finished:
//! `Row` (move along the row) → "row ends here" → `Closing` (the last tile of the row is
//! captured as soon as the view is still) → `Down` (step to the next row) → first capture of the
//! new row → `Row` in the opposite direction. "Mapping finished" ends the map the same way
//! (closing capture if there is new area, then nothing more to do).

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
    /// Row (or map) end given: hold still so the last tile can be captured.
    Closing(Then),
    /// Stepping down to the next row.
    Down,
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

    /// The row ends here. `needs_closing`: there is new area since the last tile, so one more
    /// tile must be captured.
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

    /// The map is complete (from a row, a closing or the step down).
    pub fn finish(&mut self, needs_closing: bool) {
        if self.finished {
            return;
        }
        if needs_closing && !matches!(self.phase, Phase::Closing(Then::Finish)) {
            self.phase = Phase::Closing(Then::Finish);
        } else if !needs_closing {
            self.phase = Phase::Row;
            self.finished = true;
        }
    }

    /// Flip the direction of the rows.
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
                Then::StepDown => {
                    self.phase = Phase::Down;
                    self.rows_done += 1;
                }
                Then::Finish => {
                    self.phase = Phase::Row;
                    self.finished = true;
                }
            }
        }
    }

    /// A tile was captured: completes a closing, or starts the new row (direction flips).
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
        g.end_row(true); // ignored outside of a row
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

        let mut g = Guide::default();
        g.end_row(false);
        g.finish(false); // from the step down
        assert!(g.finished);
        g.resume();
        assert!(!g.finished);
        assert_eq!(g.heading(), Some((1.0, 0.0)));
    }

    #[test]
    fn reverse_flips_row_direction() {
        let mut g = Guide::default();
        g.reverse();
        assert_eq!(g.heading(), Some((-1.0, 0.0)));
    }
}
