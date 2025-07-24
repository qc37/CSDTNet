# Enhanced Point Cloud Representation via Cross-Scale Dual-Transformer Network (CSDTNet)
 This study introduces a novel architecture, the Cross- Scale Dual-Transformer Network (CSDTNet), designed to effectively extract and aggregate features at multiple scales.

 
## Requirements
python >= 3.7, pytorch >= 1.6, h5py, scikit-learn and pip install pointnet2_ops_lib/.


## Data Preparation

### ModelNet40
You can download the dataset at https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip.

### ScanObjectNN 
You can download the dataset at https://hkust-vgd.github.io/scanobjectnn.

### ShapeNetPart
You can download the dataset at https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip.

### S3DIS 
You can download the dataset at http://buildingparser.stanford.edu/dataset.html.


## Run    
python main.py --exp_name=train --num_points=1024 --use_sgd=True --batch_size 32 --epochs 250 --lr 0.0001 

python main.py --exp_name=test --num_points=1024 --use_sgd=True --eval=True --model_path=checkpoints/best/models/model.t7 --test_batch_size 8
