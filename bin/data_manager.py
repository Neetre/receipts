import os
import sqlite3


class SQLiteDBManager:
    def __init__(self, db_name="../data/user_registration.db"):
        self.db_name = db_name

    def _connect(self):
        """Establish a new SQLite database connection and return connection and cursor."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            return conn, cursor
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            raise

    def _close(self, conn):
        """Close the SQLite database connection."""
        if conn:
            try:
                conn.close()
            except sqlite3.Error as e:
                print(f"Error closing database connection: {e}")

    def create_user_table(self):
        """Create a table for user registration."""
        conn, cursor = self._connect()
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL
                )
            ''')
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error creating user table: {e}")
        finally:
            self._close(conn)

    def insert_user(self, username, email, password):
        """Insert a new user into the users table."""
        conn, cursor = self._connect()
        try:
            cursor.execute('''
                INSERT INTO users (username, email, password)
                VALUES (?, ?, ?)
            ''', (username, email, password))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error inserting user: {e}")
        finally:
            self._close(conn)

    def fetch_all_users(self):
        """Fetch all users from the users table."""
        conn, cursor = self._connect()
        try:
            cursor.execute('SELECT * FROM users')
            return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error fetching all users: {e}")
            return []  # Return empty list or re-raise
        finally:
            self._close(conn)
    
    def fetch_user(self, username: str) -> tuple | None:
        """Fetch a single user by username."""
        conn, cursor = self._connect()
        try:
            cursor.execute("SELECT * FROM users WHERE username= ?", (username,))
            return cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error fetching user {username}: {e}")
            return None  # Return None or re-raise
        finally:
            self._close(conn)
    
    def remove_user(self, username: str):   
        """Remove a user by username."""
        conn, cursor = self._connect()
        try:
            cursor.execute("DELETE FROM users WHERE username= ?", (username,))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error removing user {username}: {e}")
        finally:
            self._close(conn)


if __name__ == "__main__":
    db_manager = SQLiteDBManager()
    db_manager.create_user_table()

    users = db_manager.fetch_all_users()
    for user in users:
        print(user)

    # db_manager.remove_user("admin")
    # db_manager.insert_user("admin", "admin@gmail.com", "a")
