import sys
sys.path.append(".")
from attacks import *
from datasets import *
from models import *
import time
from matplotlib import pyplot

if __name__ == '__main__':
    dataset = CIFAR10Dataset()
    model = CIFAR10_densenet40(rel_path='./')
    model1 = CIFAR10_final_model()
    X_test, Y_test, Y_test_target_ml, Y_test_target_ll = get_data_subset_with_systematic_attack_labels(dataset=dataset,
                                                                                                       model=model,
                                                                                                       balanced=True,
                                                                                                       num_examples=100)

    bim = Attack_BasicIterativeMethod(eps=0.008, eps_iter=0.0012, nb_iter=4)
    time_start = time.time()
    X_test_adv = bim.attack(model, X_test, Y_test)
    dur_per_sample = (time.time() - time_start) / len(X_test_adv)

    for i in range(24):
    # plot raw pixel data
      pyplot.imshow(X_test[i])
      pyplot.show()
      pyplot.imshow(X_test_adv[i])
      pyplot.show()

    # Evaluate the adversarial examples.
    print("\n---Statistics of BIM Attack on CIFAR10_densenet40 (%f seconds per sample)" % dur_per_sample)
    evaluate_adversarial_examples(X_test=X_test, Y_test=Y_test,
                                  X_test_adv=X_test_adv, Y_test_adv_pred=model.predict(X_test_adv),
                                  Y_test_target=Y_test, targeted=False)

    # Evaluate the adversarial examples.
    print("\n---Statistics of BIM Attack on CIFAR10-Final-Model (%f seconds per sample)" % dur_per_sample)
    evaluate_adversarial_examples(X_test=X_test, Y_test=Y_test,
                                  X_test_adv=X_test_adv, Y_test_adv_pred=model1.predict(X_test_adv),
                                  Y_test_target=Y_test, targeted=False)