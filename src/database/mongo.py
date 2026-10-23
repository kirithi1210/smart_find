import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "smartfind_db")

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

async def ping_db():
    try:
        await client.admin.command('ping')
        print("Connected to MongoDB successfully!")
    except Exception as e:
        print(f"MongoDB Connection error: {e}")