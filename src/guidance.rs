//! Serpentine guidance: which way to move next, row ends and the end of the map.
//!
//! `Row` (move along the row) → board edge ahead → `Closing` (the last tile of the row is
//! captured as soon as the view is still) → `Down` (step to the next row) → first capture of the
//! new row → `Row` in the opposite direction. While stepping down, an edge below means the
//! board is done: `Closing` (last tile, if any new area) → finished.
//!
//! Edges are acted upon automatically; the last automatic step can be reverted ("not the edge")
//! until the next row starts. Manual end-of-row / finish remain as fallbacks.

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
    /// Row (or map) end reached: hold still so the last tile can be captured.
    Closing(Then),
    /// Stepping down to the next row.
    Down,
}

/// What a board edge in the heading direction means right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeMeaning {
    /// Moving along a row: the row is over.
    EndOfRow,
    /// Stepping down: there is no row below, the map is complete.
    EndOfMap,
}

/// An automatic step the user can still revert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoStep {
    RowEnded,
    MapFinished,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Snapshot {
    row_dir: f64,
    phase: Phase,
    rows_done: usize,
    finished: bool,
}

#[derive(Debug, Clone)]
pub struct Guide {
    /// +1 = rows run to the right, -1 = to the left.
    pub row_dir: f64,
    pub phase: Phase,
    /// Rows completed so far.
    pub rows_done: usize,
    pub finished: bool,
    /// Last automatic step and the state before it.
    undo: Option<(AutoStep, Snapshot)>,
}

impl Default for Guide {
    fn default() -> Self {
        Self { row_dir: 1.0, phase: Phase::Row, rows_done: 0, finished: false, undo: None }
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

    /// What an edge ahead would mean now (`None`: nothing to watch for).
    pub fn edge_meaning(&self) -> Option<EdgeMeaning> {
        if self.finished {
            return None;
        }
        match self.phase {
            Phase::Row => Some(EdgeMeaning::EndOfRow),
            Phase::Down => Some(EdgeMeaning::EndOfMap),
            Phase::Closing(_) => None,
        }
    }

    /// The automatic step that "not the edge" would revert.
    pub fn auto_step(&self) -> Option<AutoStep> {
        self.undo.map(|(s, _)| s)
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot { row_dir: self.row_dir, phase: self.phase, rows_done: self.rows_done, finished: self.finished }
    }

    /// The detector saw the board edge ahead: end the row or the map, whichever applies.
    /// `needs_closing`: there is new area since the last tile, so one more tile must be captured.
    pub fn edge_reached(&mut self, needs_closing: bool) -> Option<AutoStep> {
        let before = self.snapshot();
        let step = match self.edge_meaning()? {
            EdgeMeaning::EndOfRow => {
                self.do_end_row(needs_closing);
                AutoStep::RowEnded
            }
            EdgeMeaning::EndOfMap => {
                self.do_finish(needs_closing);
                AutoStep::MapFinished
            }
        };
        self.undo = Some((step, before));
        Some(step)
    }

    /// "Not the edge": go back to before the last automatic step. Returns whether there was one.
    pub fn revert_auto(&mut self) -> bool {
        let Some((_, s)) = self.undo.take() else { return false };
        self.row_dir = s.row_dir;
        self.phase = s.phase;
        self.rows_done = s.rows_done;
        self.finished = s.finished;
        true
    }

    /// Manual: the row ends here.
    pub fn end_row(&mut self, needs_closing: bool) {
        self.undo = None;
        self.do_end_row(needs_closing);
    }

    fn do_end_row(&mut self, needs_closing: bool) {
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

    /// Manual: the map is complete (from a row, or from the step down).
    pub fn finish(&mut self, needs_closing: bool) {
        self.undo = None;
        self.do_finish(needs_closing);
    }

    fn do_finish(&mut self, needs_closing: bool) {
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

    /// Manual override: flip the direction of the rows.
    pub fn reverse(&mut self) {
        self.row_dir = -self.row_dir;
    }

    /// Resume mapping after a finish.
    pub fn resume(&mut self) {
        self.undo = None;
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
                self.undo = None; // the new row has started: too late to revert
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
        assert_eq!(g.edge_reached(true), Some(AutoStep::RowEnded));
        assert_eq!(g.phase, Phase::Closing(Then::StepDown));
        assert_eq!(g.heading(), Some((1.0, 0.0))); // still holding the row direction
        assert_eq!(g.edge_meaning(), None); // nothing to watch while capturing the last tile
        g.tile_captured(); // the closing capture
        assert_eq!(g.phase, Phase::Down);
        assert_eq!(g.rows_done, 1);
        assert_eq!(g.heading(), Some((0.0, 1.0)));
        assert_eq!(g.edge_meaning(), Some(EdgeMeaning::EndOfMap));
        g.tile_captured(); // first tile of the next row
        assert_eq!(g.phase, Phase::Row);
        assert_eq!(g.heading(), Some((-1.0, 0.0)));
        assert_eq!(g.auto_step(), None);
    }

    #[test]
    fn edge_without_new_area_skips_closing() {
        let mut g = Guide::default();
        g.edge_reached(false);
        assert_eq!(g.phase, Phase::Down);
        assert_eq!(g.edge_reached(false), Some(AutoStep::MapFinished));
        assert!(g.finished);
        assert_eq!(g.heading(), None);
        assert_eq!(g.edge_meaning(), None);
    }

    #[test]
    fn not_the_edge_reverts_the_last_automatic_step() {
        let mut g = Guide::default();
        g.edge_reached(true);
        g.tile_captured(); // closing capture done, now stepping down
        assert!(g.revert_auto());
        assert_eq!(g.phase, Phase::Row);
        assert_eq!(g.rows_done, 0);
        assert_eq!(g.heading(), Some((1.0, 0.0)));
        assert!(!g.revert_auto()); // only once

        let mut g = Guide::default();
        g.edge_reached(false);
        g.edge_reached(false); // finished from the step down
        assert_eq!(g.auto_step(), Some(AutoStep::MapFinished));
        assert!(g.revert_auto());
        assert_eq!(g.phase, Phase::Down);
        assert!(!g.finished);
    }

    #[test]
    fn manual_steps_are_not_revertible() {
        let mut g = Guide::default();
        g.end_row(false);
        assert_eq!(g.phase, Phase::Down);
        assert!(!g.revert_auto());
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
        g.finish(false);
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
