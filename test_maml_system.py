from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import MAMLFewShotClassifier
from utils.parser_utils import get_args

args, device = get_args()
model = MAMLFewShotClassifier(args=args, device=device)
data = MetaLearningSystemDataLoader
maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
test_loss_mean, test_loss_std, val_c_index = maml_system.test_experiment(args.test_model)

with open("TEST_RESULTS_11_02_2021.txt", "a+") as f:
    f.write("{},{},{},{}\n".format(args.test_path, test_loss_mean, test_loss_std, val_c_index))


