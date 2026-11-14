from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

EDITION = "bnup"
DEFAULT_GRADES = ("1", "2", "3", "4", "5", "6")


def derive_semester(unit: int, total_units: int) -> int:
    if total_units <= 0:
        return 1
    if total_units <= 2:
        return unit
    return 1 if unit <= math.ceil(total_units / 2) else 2


def _strand_from_kp(kp_id: str) -> str:
    parts = kp_id.split("-")
    return parts[2] if len(parts) >= 4 and parts[2] else ""


class BnupCorpus:
    def __init__(self, data: dict[str, Any]):
        if not isinstance(data, dict):
            raise ValueError("BNUP corpus must be a JSON object")
        grades = data.get("grades")
        if not isinstance(grades, dict) or not grades:
            raise ValueError("BNUP corpus missing 'grades' object")
        self.edition = str(data.get("edition", "beishi"))
        self.subject = str(data.get("subject", "math"))
        self.name = str(data.get("name", ""))
        self._grades = grades
        logger.info("BnupCorpus loaded: edition=%s grades=%s", self.edition, ",".join(sorted(grades)))

    def lessons(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for grade_str, grade_data in self._grades.items():
            try:
                grade = int(grade_str)
            except (TypeError, ValueError):
                logger.warning("skip invalid grade key %r", grade_str)
                continue
            units = (grade_data or {}).get("units", []) if isinstance(grade_data, dict) else []
            total = len(units)
            for u in units:
                if not isinstance(u, dict):
                    continue
                unit_no = int(u.get("unit", 0))
                unit_title = str(u.get("title", ""))
                semester = derive_semester(unit_no, total)
                for les in u.get("lessons", []) or []:
                    if not isinstance(les, dict):
                        continue
                    lesson_no = int(les.get("lesson", 0))
                    title = str(les.get("title", ""))
                    topic = str(les.get("topic", ""))
                    kp_ids = [str(k) for k in les.get("knowledge_point_ids", []) if k]
                    visual_type = str(les.get("visual_type", ""))
                    misconceptions = list(les.get("common_misconceptions", []) or [])
                    doc_id = f"bnup_g{grade}_u{unit_no}_l{lesson_no}"
                    doc_path = f"bnup/g{grade}/u{unit_no}/l{lesson_no}"
                    doc_name = f"G{grade}U{unit_no}L{lesson_no} {title}".strip()
                    content = f"{grade}年级 {unit_title} {title} {topic}".strip()
                    meta: dict[str, Any] = {
                        "edition": EDITION,
                        "grade": grade,
                        "semester": semester,
                        "unit": unit_no,
                        "lesson": lesson_no,
                        "title": title,
                        "topic": topic,
                        "knowledge_point_ids": kp_ids,
                        "visual_type": visual_type,
                        "common_misconceptions": misconceptions,
                    }
                    docs.append(
                        {
                            "doc_id": doc_id,
                            "doc_path": doc_path,
                            "doc_name": doc_name,
                            "doc_type": "bnup_lesson",
                            "content": content,
                            "metadata": meta,
                        }
                    )
        logger.info("BnupCorpus parsed %d lessons", len(docs))
        return docs

    def knowledge_graph(self) -> dict[str, Any]:
        grades_out: dict[str, Any] = {}
        kp_index: dict[str, Any] = {}
        for grade_str, grade_data in self._grades.items():
            try:
                grade = int(grade_str)
            except (TypeError, ValueError):
                continue
            units = (grade_data or {}).get("units", []) if isinstance(grade_data, dict) else []
            total = len(units)
            units_out = []
            for u in units:
                if not isinstance(u, dict):
                    continue
                unit_no = int(u.get("unit", 0))
                semester = derive_semester(unit_no, total)
                lessons_out = []
                for les in u.get("lessons", []) or []:
                    if not isinstance(les, dict):
                        continue
                    lesson_no = int(les.get("lesson", 0))
                    kp_ids = [str(k) for k in les.get("knowledge_point_ids", []) if k]
                    lessons_out.append(
                        {
                            "lesson": lesson_no,
                            "title": str(les.get("title", "")),
                            "topic": str(les.get("topic", "")),
                            "knowledge_point_ids": kp_ids,
                        }
                    )
                    for kp in kp_ids:
                        if kp not in kp_index:
                            kp_index[kp] = {
                                "grade": grade,
                                "strand": _strand_from_kp(kp),
                                "unit": unit_no,
                                "lesson": lesson_no,
                                "title": str(les.get("title", "")),
                                "topic": str(les.get("topic", "")),
                                "semester": semester,
                            }
                units_out.append(
                    {
                        "unit": unit_no,
                        "title": str(u.get("title", "")),
                        "semester": semester,
                        "lessons": lessons_out,
                    }
                )
            grades_out[str(grade)] = {"grade": grade, "units": units_out}
        return {
            "edition": self.edition,
            "subject": self.subject,
            "name": self.name,
            "grades": grades_out,
            "knowledge_points": kp_index,
            "knowledge_point_count": len(kp_index),
        }

    def stats(self) -> dict[str, Any]:
        lessons = self.lessons()
        per_grade: dict[str, Any] = {}
        per_semester: dict[str, Any] = {}
        kp_count: dict[str, int] = {}
        for d in lessons:
            m = d["metadata"]
            g = str(m["grade"])
            per_grade.setdefault(g, {"grade": m["grade"], "units": set(), "lessons": 0})
            per_grade[g]["units"].add(m["unit"])
            per_grade[g]["lessons"] += 1
            sk = f"g{g}_s{m['semester']}"
            per_semester.setdefault(sk, 0)
            per_semester[sk] += 1
            for kp in m["knowledge_point_ids"]:
                kp_count[kp] = kp_count.get(kp, 0) + 1
        for g in per_grade:
            per_grade[g]["units"] = len(per_grade[g]["units"])
        return {
            "edition": EDITION,
            "total_lessons": len(lessons),
            "per_grade": per_grade,
            "per_semester": per_semester,
            "knowledge_point_count": len(kp_count),
            "knowledge_points": kp_count,
        }
