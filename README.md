# TourPilot
In this work, we presents TourPilot, an innovative indoor tour guidance system powered by a Large Language Model (LLM) that integrates pathfinding, adaptive tour planning, and real-time navigation into a unified autonomous framework. To support LLM-driven spatial reasoning, TourPilot introduces an Indoor Entity Graph, which encodes semantic entities and spatial relationships in complex environments. Based on this, we propose a State-Driven Dynamic Planning framework that generates personalized tour plans in real time. For execution, we develop a Narrated Tour Vision-and-Language Navigation method, enabling the LLM to interpret multimodal inputs and perform navigation and tour narration without infrastructure-based positioning. The system uses instruction tuning to align LLM behavior with tour guidance functions, delivering performance comparable to that of human guides. We validate TourPilot in three real-world scenarios: CF Market Mall, Studio Bell Museum, and an art studio. A quantitative user study evaluates TourPilot’s performance across standard LLM system metrics, highlighting its potential as a next-generation solution for indoor tour guidance. 

This repository provides an indoor navigation case in an art studio, serving as a demonstration of our method.

 ![SpatialGPT](framework.png).

## Installation
1. Matterport3D installation instruction: [here](https://github.com/peteanderson80/Matterport3DSimulator). 
2. Install requirements:
```setup
conda create -n SpatialGPT python=3.10
conda activate SpatialGPT
pip install -r requirements.txt
```

## Data Preparation
1. To accelerate simulation, observation images should be pre-collected from the simulator. You can use your own saved images or use the [RGB_Observations.zip](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jadge_connect_hku_hk/Eq00RV04jXpNkwqowKh5mYABBTqBG1U2RXgQ7FvaGweJOQ?e=rL1d6p)  pre-collected in prior research work.

2. For the demonstration scan, once the above datasets setting is complete, move the file TourPilot_scene_processed.json from current directory to the new built 'datasets/R2R/annotations/' directory.

## OpenAI API key
Fill in your API key at Line 12 of the file: GPT/api.py.

## Run TourPilot
Ensure that the PYTHONPATH environment variable includes all necessary module directories (e.g., the project root) so that Python can locate internal packages like vln module.

Then run:
```bash
bash scripts/gpt4o.sh
```

Note that you should modify the following part in gpt4o.sh to set the path to your observation images, the split you want to test, etc.

```bash
--root_dir ${DATA_ROOT}
--img_root /path/to/images
--split TourPilot_scene_processed
--end 1  # the number of cases to be tested
--output_dir ${outdir}
--max_action_len 15
--save_pred
--stop_after 3
--llm gpt-4o
--response_format json
--max_tokens 4096
```

