python dynamicBA/eval_pose.py --base_path ./data/sintel --experiment_name dynamicBA --model dynamicBA

python dynamicBA/eval_pose.py --base_path ./data/sintel --experiment_name da3_no_inpaint --model da3_no_inpaint

python dynamicBA/eval_pose.py --base_path ./data/sintel --experiment_name da3_inpaint --model da3_inpaint



python dynamicBA/eval_depth.py --base_path ./data/sintel --experiment_name dynamicBA --model dynamicBA

python dynamicBA/eval_depth.py --base_path ./data/sintel --experiment_name da3_no_inpaint --model da3_no_inpaint

python dynamicBA/eval_depth.py --base_path ./data/sintel --experiment_name da3_inpaint --model da3_inpaint



python dynamicBA/eval_depth_masked.py --base_path ./data/sintel --experiment_name dynamicBA_masked --model dynamicBA

python dynamicBA/eval_depth_masked.py --base_path ./data/sintel --experiment_name da3_no_inpaint_masked --model da3_no_inpaint

python dynamicBA/eval_depth_masked.py --base_path ./data/sintel --experiment_name da3_inpaint_masked --model da3_inpaint