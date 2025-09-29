from flask import Flask, request, jsonify, abort
from functools import wraps
import os
import jwt
import datetime
import uuid
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from dotenv import load_dotenv
import google.generativeai as genai
import json

# --------- ENV + Setup ---------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/ai_quiz_db")
JWT_SECRET = os.getenv("JWT_SECRET", "change_this_secret")
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 60 * 60 * 24
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

# MongoDB
client = MongoClient(MONGO_URI)
db = client.get_default_database()
users_col = db.users
quizzes_col = db.quizzes
submissions_col = db.submissions

# Gemini init
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# --------- JWT Helpers ---------
def generate_jwt(user_id: str, username: str):
    payload = {
        "sub": user_id,
        "username": username,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

def decode_jwt(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        abort(401, description="Token expired")
    except jwt.InvalidTokenError:
        abort(401, description="Invalid token")

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", None)
        if not auth:
            return jsonify({"error": "Authorization header missing"}), 401
        parts = auth.split()
        if parts[0].lower() != "bearer" or len(parts) != 2:
            return jsonify({"error": "Authorization header must be: Bearer <token>"}), 401
        token = parts[1]
        payload = decode_jwt(token)
        request.user = {"user_id": payload.get("sub"), "username": payload.get("username")}
        return f(*args, **kwargs)
    return decorated

# --------- Gemini AI Helpers ---------
def ai_generate_questions(grade: int, subject: str, total_questions: int, difficulty: str):
    prompt = f"""
    Generate {total_questions} multiple-choice questions for Grade {grade} {subject}.
    Difficulty: {difficulty}.
    Respond ONLY with a valid JSON array of objects with fields: questionId, text, choices, correctAnswer, difficulty.
    """
    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()

        # Try JSON parse
        try:
            questions = json.loads(raw)
        except Exception as je:
            print("Parsing error:", je, "Raw response:", raw)
            # fallback: make dummy questions
            questions = [
                {
                    "questionId": str(uuid.uuid4()),
                    "text": f"Dummy Q{i+1}: 2+2?",
                    "choices": ["A. 3", "B. 4", "C. 5", "D. 6"],
                    "correctAnswer": "B",
                    "difficulty": difficulty
                } for i in range(total_questions)
            ]
        return questions

    except Exception as e:
        print("Gemini error:", e)
        # fallback: return dummy set
        return [
            {
                "questionId": str(uuid.uuid4()),
                "text": f"Fallback Q{i+1}: 5+3?",
                "choices": ["A. 7", "B. 8", "C. 9", "D. 10"],
                "correctAnswer": "B",
                "difficulty": difficulty
            } for i in range(total_questions)
        ]

def ai_generate_hint(question_text: str):
    prompt = f"Provide a short helpful hint for the question: {question_text}"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except:
        return "Think carefully and eliminate wrong choices."

def ai_generate_suggestions(mistakes: list):
    mistake_text = ", ".join([m.get("questionId", "") for m in mistakes]) or "no mistakes"
    prompt = f"""
    The user made mistakes in: {mistake_text}.
    Suggest 2 improvement tips to help them improve.
    """
    try:
        response = gemini_model.generate_content(prompt)
        print(response.text.strip())
        return response.text.strip().split("\n")[:2]
    except:
        return ["Review mistakes and retry.", "Practice more questions."]

# --------- Adaptive difficulty helpers ---------

def compute_user_performance(user_id: str, subject: str = None):
    """
    Compute a simple accuracy metric from past submissions for the user optionally filtered by subject.
    Returns a float between 0 and 1.
    """
    query = {"userId": user_id}
    if subject:
        query["subject"] = subject
    cursor = submissions_col.find(query, {"score": 1, "maxScore": 1})
    total = 0
    obtained = 0
    for doc in cursor:
        total += doc.get("maxScore", 0)
        obtained += doc.get("score", 0)
    if total == 0:
        return None  # no history
    return obtained / total


def decide_balance_from_performance(perf: float, total_questions: int):
    """
    perf None -> balanced default
    perf < 0.5 -> easier
    perf 0.5-0.8 -> medium
    perf > 0.8 -> harder
    Returns distribution dict
    """
    if perf is None:
        # default balanced
        return {"EASY": int(total_questions * 0.4), "MEDIUM": int(total_questions * 0.4), "HARD": total_questions - int(total_questions * 0.4) - int(total_questions * 0.4)}
    if perf < 0.5:
        return {"EASY": int(total_questions * 0.6), "MEDIUM": int(total_questions * 0.3), "HARD": total_questions - int(total_questions * 0.6) - int(total_questions * 0.3)}
    if perf < 0.8:
        return {"EASY": int(total_questions * 0.3), "MEDIUM": int(total_questions * 0.5), "HARD": total_questions - int(total_questions * 0.3) - int(total_questions * 0.5)}
    return {"EASY": int(total_questions * 0.2), "MEDIUM": int(total_questions * 0.4), "HARD": total_questions - int(total_questions * 0.2) - int(total_questions * 0.4)}

# --------- Endpoints ---------

# --------- Auth Endpoints ---------
@app.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username, password = data.get("username"), data.get("password")
    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    if users_col.find_one({"username": username}):
        return jsonify({"error": "Username already exists"}), 400
    user_doc = {"username": username, "password": password, "createdAt": datetime.datetime.utcnow()}
    res = users_col.insert_one(user_doc)
    return jsonify({"message": "User registered", "userId": str(res.inserted_id)})

@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username, password = data.get("username"), data.get("password")
    user = users_col.find_one({"username": username})
    if not user or user.get("password") != password:
        return jsonify({"error": "Invalid credentials"}), 401
    token = generate_jwt(str(user["_id"]), username)
    return jsonify({"token": token, "userId": str(user["_id"])})

# --------- Quiz Endpoints ---------
@app.route("/quiz/generate", methods=["POST"])
@require_auth
def generate_quiz():
    payload = request.get_json() or {}
    grade = int(payload["grade"])
    subject = payload["Subject"]
    total_questions = int(payload["TotalQuestions"])
    max_score = int(payload["MaxScore"])
    difficulty = payload.get("Difficulty", "MEDIUM")

    questions = ai_generate_questions(grade, subject, total_questions, difficulty)
    # print(questions)
    if not questions:
        return jsonify({"error": "Failed to generate quiz"}), 500

    quiz_doc = {
        "grade": grade, "subject": subject, "totalQuestions": total_questions,
        "maxScore": max_score, "difficulty": difficulty,
        "questions": questions, "createdBy": request.user["user_id"],
        "createdAt": datetime.datetime.utcnow(),
    }
    res = quizzes_col.insert_one(quiz_doc)
    quiz_id = str(res.inserted_id)

    public_questions = [{"questionId": q["questionId"], "text": q["text"], "choices": q["choices"], "difficulty": q["difficulty"]} for q in questions]
    return jsonify({"quizId": quiz_id, "questions": public_questions})

@app.route("/quiz/<quiz_id>/hint", methods=["POST"])
@require_auth
def get_hint(quiz_id):
    data = request.get_json() or {}
    qid = data.get("questionId")
    quiz = quizzes_col.find_one({"_id": ObjectId(quiz_id)})
    question = next((q for q in quiz.get("questions", []) if q["questionId"] == qid), None)
    hint = ai_generate_hint(question["text"])
    return jsonify({"questionId": qid, "hint": hint})

@app.route("/quiz/submit", methods=["POST"])
@require_auth
def submit_quiz():
    data = request.get_json() or {}
    quiz_id, responses = data.get("quizId"), data.get("responses", [])
    quiz = quizzes_col.find_one({"_id": ObjectId(quiz_id)})
    answer_map = {q["questionId"]: q["correctAnswer"] for q in quiz["questions"]}

    score, mistakes, detailed = 0, [], []
    per_score = quiz["maxScore"] / len(answer_map)
    for resp in responses:
        qid, user_resp = resp["questionId"], resp["userResponse"]
        correct = answer_map.get(qid)
        if user_resp == correct:
            score += per_score
            detailed.append({"questionId": qid, "userResponse": user_resp, "isCorrect": True})
        else:
            mistakes.append({"questionId": qid, "userResponse": user_resp, "correctAnswer": correct})
            detailed.append({"questionId": qid, "userResponse": user_resp, "isCorrect": False, "correctAnswer": correct})

    suggestions = ai_generate_suggestions(mistakes)
    submission_doc = {
        "quizId": quiz_id, "userId": request.user["user_id"],
        "responses": detailed, "mistakes": mistakes,
        "score": round(score, 2), "maxScore": quiz["maxScore"],
        "subject": quiz["subject"], "grade": quiz["grade"],
        "createdAt": datetime.datetime.utcnow(),
    }
    res = submissions_col.insert_one(submission_doc)
    return jsonify({"submissionId": str(res.inserted_id), "score": round(score, 2), "maxScore": quiz["maxScore"], "suggestions": suggestions})



@app.route("/quiz/history", methods=["GET"]) 
@require_auth
def quiz_history():
    # Filters supported via query parameters: grade, subject, minScore, maxScore, from, to, userId(optional)
    args = request.args
    query = {}
    # allow admin-like access by specifying userId, otherwise default to own
    user_filter = args.get("userId") or request.user["user_id"]
    query["userId"] = user_filter

    if args.get("grade"):
        try:
            query["grade"] = int(args.get("grade"))
        except:
            pass
    if args.get("subject"):
        query["subject"] = args.get("subject")
    if args.get("minScore"):
        try:
            query["score"] = {"$gte": float(args.get("minScore"))}
        except:
            pass
    if args.get("from") or args.get("to"):
        date_query = {}
        if args.get("from"):
            try:
                date_from = datetime.datetime.strptime(args.get("from"), "%d/%m/%Y")
                date_query["$gte"] = date_from
            except:
                pass
        if args.get("to"):
            try:
                date_to = datetime.datetime.strptime(args.get("to"), "%d/%m/%Y")
                # include end of day
                date_to = date_to + datetime.timedelta(days=1)
                date_query["$lte"] = date_to
            except:
                pass
        if date_query:
            query["createdAt"] = date_query

    cursor = submissions_col.find(query).sort("createdAt", DESCENDING).limit(200)
    results = []
    for doc in cursor:
        results.append({
            "submissionId": str(doc.get("_id")),
            "quizId": doc.get("quizId"),
            "score": doc.get("score"),
            "maxScore": doc.get("maxScore"),
            "mistakesCount": len(doc.get("mistakes", [])),
            "createdAt": doc.get("createdAt").isoformat(),
            "subject": doc.get("subject"),
            "grade": doc.get("grade"),
        })

    return jsonify({"history": results})


@app.route("/quiz/<quiz_id>/retry", methods=["POST"]) 
@require_auth
def retry_quiz(quiz_id):
    # Allow user to retry: we simply fetch quiz and return the public questions again
    # A retry is a fresh submission when user posts to /quiz/submit
    quiz = quizzes_col.find_one({"_id": ObjectId(quiz_id)})
    if not quiz:
        return jsonify({"error": "Quiz not found"}), 404
    public_questions = [{"questionId": q["questionId"], "text": q["text"], "choices": q["choices"], "difficulty": q["difficulty"]} for q in quiz.get("questions", [])]
    return jsonify({"quizId": quiz_id, "questions": public_questions})


@app.route("/quiz/<quiz_id>/details", methods=["GET"]) 
@require_auth
def quiz_details(quiz_id):
    # Return quiz meta (not correct answers)
    quiz = quizzes_col.find_one({"_id": ObjectId(quiz_id)})
    if not quiz:
        return jsonify({"error": "Quiz not found"}), 404
    resp = {
        "quizId": quiz_id,
        "grade": quiz.get("grade"),
        "subject": quiz.get("subject"),
        "totalQuestions": quiz.get("totalQuestions"),
        "maxScore": quiz.get("maxScore"),
        "difficulty": quiz.get("difficulty"),
        "createdBy": quiz.get("createdBy"),
        "createdAt": quiz.get("createdAt").isoformat(),
    }
    return jsonify(resp)


# Health check
@app.route("/health", methods=["GET"]) 
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
