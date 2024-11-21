import sqlite3

def clear_data():
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    
    #cursor.execute('DELETE FROM users')
    cursor.execute('DELETE FROM chatbot')
    
    conn.commit()  
    conn.close()   

    print("All data cleared successfully!")

if __name__ == "__main__":
    clear_data()
