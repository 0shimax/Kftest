import kfp
from kfp import dsl


# GCSからファイルをダウンロードするComponentを生成する関数
def gcs_download_op(url):
    # dsl.ContainerOpに実行するDocker Imageやコンテナで実行するコマンドや
    # 引数を指定することでComponentを生成する
    return dsl.ContainerOp(
        name='GCS - Download',
        image='google/cloud-sdk:272.0.0',
        command=['sh', '-c'],
        arguments=['gsutil cat $0 | tee $1', url, '/tmp/results.txt'],
        # 下流(downstream)のタスクにデータを受け渡したいときは、
        # ファイルに書き出してそのパスをfile_outputsに渡すと値を渡せる
        file_outputs={
            'data': '/tmp/results.txt',
        }
    )


# 上流(upstream)でダウンロードしたファイルの中身をechoするComponentを生成する関数
def echo2_op(text1, text2):
    return dsl.ContainerOp(
        name='echo',
        image='library/bash:4.4.23',
        command=['sh', '-c'],
        arguments=['echo "Text 1: $0"; echo "Text 2: $1"', text1, text2]
    )


# Pipelineを構築する関数にはdsl.pipelineデコレータを付与する
@dsl.pipeline(
  name='Parallel pipeline',
  description='Download two messages in parallel and prints the concatenated result.'
)
def download_and_join(
    url1='gs://ml-pipeline-playground/shakespeare1.txt',
    url2='gs://ml-pipeline-playground/shakespeare2.txt'
):
    """A three-step pipeline with first two running in parallel."""

    download1_task = gcs_download_op(url1)
    download2_task = gcs_download_op(url2)

    # 上流タスクのoutputを受け取ってタスクを実行
    echo_task = echo2_op(download1_task.output, download2_task.output)


if __name__ == '__main__':
    # Pipelineを定義するYAMLを生成
    kfp.compiler.Compiler().compile(download_and_join, __file__ + '.yaml')