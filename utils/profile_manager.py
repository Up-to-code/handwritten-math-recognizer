import json
import os
from datetime import datetime


class StudentProfileManager:
    def __init__(self, profiles_dir="profiles"):
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)

    def _profile_path(self, student_id):
        return os.path.join(self.profiles_dir, f"{student_id}.json")

    def get_profile(self, student_id):
        path = self._profile_path(student_id)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    def create_profile(self, student_id, name):
        profile = {
            "student_id": student_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "total_questions": 0,
            "correct_answers": 0,
            "incorrect_answers": 0,
            "grade": "N/A",
            "history": [],
        }
        with open(self._profile_path(student_id), "w") as f:
            json.dump(profile, f, indent=2)
        return profile

    def record_answer(self, student_id, question_number, correct_answer, student_answer):
        profile = self.get_profile(student_id)
        if profile is None:
            return None

        is_correct = correct_answer == student_answer

        profile["total_questions"] += 1
        if is_correct:
            profile["correct_answers"] += 1
        else:
            profile["incorrect_answers"] += 1

        total = profile["total_questions"]
        correct = profile["correct_answers"]
        percentage = (correct / total * 100) if total > 0 else 0

        if percentage >= 90:
            profile["grade"] = "A"
        elif percentage >= 80:
            profile["grade"] = "B"
        elif percentage >= 70:
            profile["grade"] = "C"
        elif percentage >= 60:
            profile["grade"] = "D"
        else:
            profile["grade"] = "F"

        profile["history"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "question_number": question_number,
                "correct_answer": correct_answer,
                "student_answer": student_answer,
                "is_correct": is_correct,
            }
        )

        with open(self._profile_path(student_id), "w") as f:
            json.dump(profile, f, indent=2)

        return {
            "is_correct": is_correct,
            "profile": profile,
            "percentage": percentage,
        }

    def list_profiles(self):
        profiles = []
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.profiles_dir, filename), "r") as f:
                    profiles.append(json.load(f))
        return profiles

    def get_leaderboard(self):
        profiles = self.list_profiles()
        leaderboard = []
        for p in profiles:
            total = p["total_questions"]
            correct = p["correct_answers"]
            pct = (correct / total * 100) if total > 0 else 0
            leaderboard.append(
                {
                    "name": p["name"],
                    "student_id": p["student_id"],
                    "correct": correct,
                    "total": total,
                    "percentage": round(pct, 1),
                    "grade": p["grade"],
                }
            )
        leaderboard.sort(key=lambda x: x["percentage"], reverse=True)
        return leaderboard
