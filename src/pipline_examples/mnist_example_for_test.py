import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.onprem as onprem


platform = 'GCP'

@dsl.pipeline(
    name='MNIST',
    description='A pipeline to test the MNIST example.'
)
def mnist_pipeline(model_export_dir='gs://kf-test1234/export',
                   project='<your project id>',
                   bucket_name='kf-test1234',
                   n_class='10',
                   pvc_name=''):
    test = _test(
            project,
            bucket_name,
            n_class,
            model_export_dir)

    steps = [test]
    for step in steps:
        if platform == 'GCP':
            step.apply(gcp.use_gcp_secret('user-gcp-sa'))
        else:
            step.apply(onprem.mount_pvc(pvc_name, 'local-storage', '/mnt'))


def _test(project,
          bucket_name,
          n_class,
          model_export_dir):
    return dsl.ContainerOp(
            name='test',
            image='gcr.io/<your project id>/wbc-model:v0.1.0',
            command=['python3', 'pipline_examples/mnist_test.py'],
            arguments=[
                    '--project', project,
                    '--bucket-name', bucket_name,
                    '--n-class', n_class,
                    '--out-dir', model_export_dir
            ],
            file_outputs={
                'MLPipeline Metrics': '/mlpipeline-metrics.json',
                'MLPipeline UI metadata': '/mlpipeline-ui-metadata.json'
            }
        )


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(mnist_pipeline, __file__ + '.tar.gz')