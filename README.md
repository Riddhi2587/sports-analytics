# Modular Sports Analytics Pipeline

A generalized computer vision framework for detecting events and tracking objects across different sports. This project uses a unified router to handle specific tasks—like counting goals or tracking players—across Basketball, Soccer, Tennis, and Fitness routines.

---

## ⚡ Quick Start

### Clone the repository

```bash
git clone https://github.com/yourusername/sports-analytics-pipeline.git
cd sports-analytics-pipeline
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file and add your Roboflow API key:

```ini
ROBOFLOW_API_KEY=your_api_key_here
```

---

## 🏃 Usage

Run the analysis using the central router `main.py`. You must specify the sport, the task, and the input video.

```bash
python main.py --sport <SPORT> --task <TASK> --video <PATH_TO_VIDEO>
```

### Supported Combinations

| Sport      | Task          | Description                                |
|------------|---------------|--------------------------------------------|
| basketball | count_goals   | Count made baskets and team scores.        |
| basketball | track_players | Track players and ball trajectories.       |
| soccer     | count_goals   | Count goals and score events.              |
| soccer     | track_players | Track players on the pitch.                |
| pullups    | count_reps    | Count exercise repetitions.                |
| tennis     | track_players | Track player movement and history.         |

### Example

Count goals in a soccer match:

```bash
python main.py --sport soccer --task count_goals --video inputs/match.mp4
```

---

## 📂 Project Structure

- `main.py`: The entry point that registers pipelines and routes commands.
- `basketball/`, `soccer/`, `tennis/`, `pullups/`: Specialized modules containing sport-specific logic and event rules.

