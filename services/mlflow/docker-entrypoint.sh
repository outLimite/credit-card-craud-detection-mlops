set -o errexit    
set -o nounset      

case "$1" in
  serve)
	python3.12 src/prepare_bucket.py && \
    mlflow server \
		  --backend-store-uri sqlite:///"$MLFLOW_HOME"/mlflow.db \
		  --default-artifact-root s3://${AWS_BUCKET_NAME}/ \
		  --artifacts-destination s3://${AWS_BUCKET_NAME}/ \
		  --workers $MLFLOW_NUM_WORKERS \
		  --host 0.0.0.0 \
		  --port $MLFLOW_PORT
    ;;
  *)
    exec "$@"
esac