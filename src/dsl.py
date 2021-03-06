import kfp
import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.onprem as onprem


platform = 'GCP'


@dsl.pipeline(
  name='WBC',
  description='A pipeline to train and test the WBC'
)
def wbc_pipline(model_export_dir='export/wbc',
                data_root='data/segmentation_WBC-master',
                metadata_file_name='Class_Labels_of_{}.csv',
                subset='Dataset1',
                project='graphic-option-220202',
                bucket_name='kf-test1234',
                n_class="5",
                resume_model='export/wbc/NFCM_model.pth',
                epochs='50',
                batch_size='32',
                pvc_name=''):
    train = _train(data_root,
                   metadata_file_name,                  
                   subset,
                   project,
                   bucket_name,
                   n_class,
                   epochs,
                   batch_size,
                   model_export_dir)  # .set_gpu_limit(1)
    # train.add_node_selector_constraint('cloud.google.com/gke-nodepool', 'gpu-pool')
    # out = train.outputs['output']
    
    test = _test(data_root,
                 metadata_file_name,
                 subset,
                 project,
                 bucket_name,
                 n_class,
                 resume_model,
                 model_export_dir)
    test.after(train)

    steps = [train, test]
    for step in steps:
        if platform == 'GCP':
            step.apply(gcp.use_gcp_secret('user-gcp-sa'))


def _train(data_root,
           metadata_file_name,
           subset,
           project,
           bucket_name,
           n_class,
           epochs,
           batch_size,
           model_export_dir):
    return dsl.ContainerOp(
            name='train',
            image='gcr.io/graphic-option-220202/wbc-model:v0.1.0',
            command=['python3', 'train.py'],
            arguments=[
                    '--data-root', data_root,
                    '--metadata-file-name', metadata_file_name,
                    '--subset', subset,
                    '--project', project,
                    '--bucket-name', bucket_name,
                    '--n-class', n_class,
                    '--batch-size', batch_size,
                    '--epochs', epochs,
                    '--out-dir', model_export_dir
            ],
            file_outputs={
                'output': '/output.txt',
            }
        )


def _test(data_root,
          metadata_file_name,
          subset,
          project,
          bucket_name,
          n_class,
          resume_model,
          model_export_dir):
    return dsl.ContainerOp(
            name='test',
            image='gcr.io/graphic-option-220202/wbc-model:v0.1.0',
            command=['python3', 'test.py'],
            arguments=[
                    '--data-root', data_root,
                    '--metadata-file-name', metadata_file_name,
                    '--subset', subset,
                    '--project', project,
                    '--bucket-name', bucket_name,
                    '--n-class', n_class,
                    '--resume-model', resume_model,
                    '--out-dir', model_export_dir
            ],
            file_outputs={
                'MLPipeline Metrics': '/mlpipeline-metrics.json',
                'MLPipeline UI metadata': '/mlpipeline-ui-metadata.json'
            }
        )


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(wbc_pipline, __file__ + '.tar.gz')