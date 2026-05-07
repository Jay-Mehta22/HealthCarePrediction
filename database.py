import sqlite3
import hashlib

# ---------- CONNECTION ----------
def create_connection():
    return sqlite3.connect("health.db", check_same_thread=False)


# ---------- HASH PASSWORD ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ---------- CREATE TABLES ----------
def create_tables():
    conn = create_connection()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        username TEXT,
        disease TEXT,
        result TEXT,
        probability REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


# ---------- CREATE USER ----------
def create_user(username, password):
    username = username.strip()
    password = hash_password(password.strip())  # 🔐 HASH HERE

    conn = create_connection()
    c = conn.cursor()

    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False


# ---------- LOGIN USER ----------
def login_user(username, password):
    username = username.strip()
    password = hash_password(password.strip())  # 🔐 HASH HERE

    conn = create_connection()
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    data = c.fetchone()

    conn.close()

    return data is not None


# ---------- SAVE HISTORY ----------
def save_history(username, disease, result, probability):
    conn = create_connection()
    c = conn.cursor()

    c.execute(
        "INSERT INTO history (username, disease, result, probability) VALUES (?, ?, ?, ?)",
        (username, disease, result, probability)
    )

    conn.commit()
    conn.close()


# ---------- GET HISTORY ----------
def get_history(username):
    conn = create_connection()
    c = conn.cursor()

    c.execute("""
        SELECT disease, result, probability, timestamp
        FROM history
        WHERE username=?
        ORDER BY timestamp
    """, (username,))

    data = c.fetchall()

    conn.close()
    return data