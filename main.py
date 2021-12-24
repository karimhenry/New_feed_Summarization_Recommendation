import uvicorn
from Recommendation.Router import app

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
