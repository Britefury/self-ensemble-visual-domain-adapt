python experiment_domainadapt_meanteacher.py --exp=mnist_usps --log_file=results_exp/log_mnist_usps_run${2}.txt --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}
python experiment_domainadapt_meanteacher.py --exp=usps_mnist --log_file=results_exp/log_usps_mnist_run${2}.txt --standardise_samples --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}

python experiment_domainadapt_meanteacher.py --exp=mnist_svhn --log_file=results_exp/log_mnist_svhn_stdaug_run${2}.txt --standardise_samples --batch_size=256 --num_epochs=300 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=svhn_mnist --log_file=results_exp/log_svhn_mnist_stdaug_run${2}.txt --standardise_samples --batch_size=256 --num_epochs=300 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=mnist_svhn --log_file=results_exp/log_mnist_svhn_run${2}.txt --standardise_samples --src_intens_flip --src_intens_scale_range=0.25:1.5 --src_intens_offset_range=-0.5:0.5 --tgt_intens_flip --tgt_intens_scale_range=0.25:1.5 --tgt_intens_offset_range=-0.5:0.5 --batch_size=256 --num_epochs=300 --device=${1}
python experiment_domainadapt_meanteacher.py --exp=svhn_mnist --log_file=results_exp/log_svhn_mnist_run${2}.txt --standardise_samples --src_intens_flip --src_intens_scale_range=0.25:1.5 --src_intens_offset_range=-0.5:0.5 --tgt_intens_flip --tgt_intens_scale_range=0.25:1.5 --tgt_intens_offset_range=-0.5:0.5 --batch_size=256 --num_epochs=300 --device=${1}

python experiment_domainadapt_meanteacher.py --exp=cifar_stl --log_file=results_exp/log_cifar_stl_run${2}.txt --standardise_samples --src_hflip --tgt_hflip --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}
python experiment_domainadapt_meanteacher.py --exp=stl_cifar --log_file=results_exp/log_stl_cifar_run${2}.txt --standardise_samples --src_hflip --tgt_hflip --batch_size=256 --cls_balance=0.0 --num_epochs=300 --epoch_size=large --device=${1}

python experiment_domainadapt_meanteacher.py --exp=syndigits_svhn --log_file=results_exp/log_syndigits_svhn_run${2}.txt --standardise_samples --batch_size=128 --cls_balance=0.0 --num_epochs=300 --device=${1}

python experiment_domainadapt_meanteacher.py --exp=synsigns_gtsrb --log_file=results_exp/log_synsigns_gtsrb_run${2}.txt --standardise_samples --batch_size=128 --cls_balance=0.0 --num_epochs=300 --device=${1}
