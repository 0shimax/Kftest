import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.onprem as onprem


platform = 'GCP'


@dsl.pipeline(
  name='WBC',
  description='A pipeline to train and test the WBC'
)
def wbc_pipline(model_export_dir='gs://kf-test1234/export',
                data_root='gs://kf-test1234/data/segmentation_WBC-master',
                metadata_file_name='Class_Labels_of_{}.csv',
                subset='Dataset1',
                n_class=5,
                resume_model='',
                epochs='50',
                batch_size='32',
                pvc_name=''):
    train = train(model_export_dir,
                  data_root,
                  metadata_file_name,
                  subset,
                  n_class,
                  resume_model,
                  epochs,
                  batch_size).set_gpu_limit(1)
    train.add_node_selector_constraint('cloud.google.com/gke-nodepool', 'gpu-pool')
    out = train.outputs['output']
    
    test = test(model_export_dir,
                data_root,
                metadata_file_name,
                subset,
                n_class,
                resume_model,
                epochs,
                batch_size)
    test.after(train)

    steps = [train, test]
    for step in steps:
        if platform == 'GCP':
            step.apply(gcp.use_gcp_secret('user-gcp-sa'))


def train(model_export_dir,
          data_root,
          metadata_file_name,
          subset,
          n_class,
          resume_model,
          epochs,
          batch_size):
    return dsl.ContainerOp(
            image='gcr.io/<your project id>/wbc-model:v0.1.0',
            command=['python3', 'train.py'],
            arguments=[
                    '--data-root', data_root,
                    '--metadata-file-name', metadata_file_name,
                    '--subset', subset,
                    '--n-class', n_class,
                    '--resume-model', resume_model,
                    '--epochs', epochs,
                    '--batch_size', batch_size,
                    '--out-dir', model_export_dir
            ],
            file_outputs={
                        'output': '/output.txt',
            }
        )


def test(model_export_dir,
         data_root,
         metadata_file_name,
         subset,
         n_class,
         resume_model,
         epochs,
         batch_size):
    return dsl.ContainerOp(
            image='gcr.io/<your project id>/wbc-model:v0.1.0',
            command=['python3', 'test.py'],
            arguments=[
                    '--data-root', data_root,
                    '--metadata-file-name', metadata_file_name,
                    '--subset', subset,
                    '--n-class', n_class,
                    '--resume-model', resume_model,
                    '--epochs', epochs,
                    '--batch_size', batch_size,
                    '--out-dir', model_export_dir
            ],
            file_outputs={
                        'MLPipeline Metrics': '/mlpipeline-metrics.json',
                        'MLPipeline UI metadata': '/mlpipeline-ui-metadata.json'
            }
        )


if __name__ == '__main__':
    compiler.Compiler().compile(clf_pipeline, __file__ + '.tar.gz')