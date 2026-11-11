"""
Attendance Reporting and Analytics CLI.

Provides:
- Detailed per-session reports (terminal table + CSV export)
- Cumulative per-student attendance percentage tracking across all sessions
- Course-level aggregate statistics
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

import db
from config import EXPORTS_DIR, DB_PATH


def print_session_report(session_id: int, export_csv: bool = True) -> None:
    """Generate detailed console summary and CSV export for a specific session."""
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?;", (session_id,))
        session = cursor.fetchone()

    if not session:
        print(f"Error: Session #{session_id} not found in database.")
        return

    records = db.get_session_attendance(session_id)
    if not records:
        print(f"Warning: No attendance records found for Session #{session_id}.")
        return

    df = pd.DataFrame(records)
    total_enrolled = len(df)
    present_df = df[df["status"] == "PRESENT"]
    absent_df = df[df["status"] == "ABSENT"]

    present_count = len(present_df)
    absent_count = len(absent_df)
    attendance_pct = (present_count / total_enrolled * 100) if total_enrolled > 0 else 0

    print("=" * 75)
    print(f"LECTURE SESSION REPORT: #{session_id}")
    print("=" * 75)
    print(f"Course         : {session['course']}")
    print(f"Date & Time    : {session['date']} at {session['start_time']}")
    print(f"Total Enrolled : {total_enrolled}")
    print(f"Present        : {present_count} ({attendance_pct:.1f}%)")
    print(f"Absent         : {absent_count} ({100 - attendance_pct:.1f}%)")
    print("-" * 75)
    print(f"{'Roll No':<10} | {'Name':<22} | {'Status':<8} | {'Frames':<7} | {'Confidence'}")
    print("-" * 75)

    for _, row in df.iterrows():
        tag = "[P]" if row["status"] == "PRESENT" else "[A]"
        print(
            f"{row['roll_no']:<10} | {row['name']:<22} | {tag} {row['status']:<4} | "
            f"{row['frames_seen']:<7} | {row['confidence']:.4f}"
        )
    print("=" * 75)

    if export_csv:
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        clean_course = "".join(c if c.isalnum() else "_" for c in session["course"])
        out_csv = EXPORTS_DIR / f"attendance_session_{session_id}_{session['date']}_{clean_course}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Exported session CSV: {out_csv}")


def print_all_sessions() -> None:
    """Print an overview table of all recorded attendance sessions."""
    sessions = db.get_all_sessions()
    print("=" * 75)
    print(f"{'ID':<4} | {'Date':<11} | {'Time':<9} | {'Course':<24} | {'Present/Total':<14} | {'Rate'}")
    print("-" * 75)

    if not sessions:
        print("  No sessions recorded yet. Run 'python run_session.py' to record a session.")

    for s in sessions:
        records = db.get_session_attendance(s["id"])
        total = len(records)
        present = sum(1 for r in records if r["status"] == "PRESENT")
        rate = f"{(present / total * 100):.1f}%" if total > 0 else "0.0%"
        ratio = f"{present}/{total}"
        print(
            f"{s['id']:<4} | {s['date']:<11} | {s['start_time']:<9} | "
            f"{s['course']:<24} | {ratio:<14} | {rate}"
        )
    print("=" * 75)


def print_cumulative_student_summary(export_csv: bool = True) -> None:
    """
    Compute cumulative attendance rates per student across all sessions.
    Shows total lectures, attended count, and overall percentage.
    """
    students = db.get_all_students()
    sessions = db.get_all_sessions()

    if not students:
        print("No students enrolled.")
        return

    if not sessions:
        print("No sessions recorded yet.")
        return

    summary_rows = []

    with db.get_connection() as conn:
        cursor = conn.cursor()
        for s in students:
            cursor.execute(
                """
                SELECT 
                    COUNT(session_id) as total_sessions,
                    SUM(CASE WHEN status = 'PRESENT' THEN 1 ELSE 0 END) as attended
                FROM attendance
                WHERE student_id = ?;
                """,
                (s["id"],),
            )
            row = cursor.fetchone()
            total_sessions = row["total_sessions"] or 0
            attended = row["attended"] or 0
            absent = total_sessions - attended
            pct = (attended / total_sessions * 100) if total_sessions > 0 else 0.0

            summary_rows.append(
                {
                    "Roll No": s["roll_no"],
                    "Name": s["name"],
                    "Total Sessions": total_sessions,
                    "Attended": attended,
                    "Absent": absent,
                    "Attendance %": round(pct, 2),
                }
            )

    df = pd.DataFrame(summary_rows)
    df.sort_values(by="Roll No", inplace=True)

    print("=" * 75)
    print("CUMULATIVE STUDENT ATTENDANCE SUMMARY")
    print("=" * 75)
    print(f"Total Sessions Held: {len(sessions)}")
    print("-" * 75)
    print(f"{'Roll No':<10} | {'Name':<22} | {'Total':<6} | {'Attended':<8} | {'Absent':<6} | {'Rate'}")
    print("-" * 75)

    for _, r in df.iterrows():
        # Highlight low attendance (< 75%)
        flag = " [LOW]" if r["Attendance %"] < 75.0 else ""
        print(
            f"{r['Roll No']:<10} | {r['Name']:<22} | {r['Total Sessions']:<6} | "
            f"{r['Attended']:<8} | {r['Absent']:<6} | {r['Attendance %']:5.1f}%{flag}"
        )
    print("=" * 75)

    if export_csv:
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = EXPORTS_DIR / f"cumulative_student_attendance_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Exported cumulative CSV: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Attendance Analytics and Report Generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--latest", action="store_true", help="Show report for the latest session")
    group.add_argument("--session", type=int, help="Show report for a specific session ID")
    group.add_argument("--sessions", action="store_true", help="List all recorded sessions")
    group.add_argument(
        "--student-summary",
        action="store_true",
        help="Compute cumulative attendance rates for every student",
    )

    args = parser.parse_args()

    if args.latest:
        sessions = db.get_all_sessions()
        if not sessions:
            print("No sessions found in database.")
            return
        latest_id = sessions[0]["id"]
        print_session_report(latest_id)
    elif args.session:
        print_session_report(args.session)
    elif args.sessions:
        print_all_sessions()
    elif args.student_summary:
        print_cumulative_student_summary()


if __name__ == "__main__":
    main()
