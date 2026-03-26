#!/usr/bin/env bash

MODEL_ID="$1"

if [ -z "$MODEL_ID" ]; then
  echo "Usage: $0 <bedrock-model-id>"
  exit 1
fi

aws pricing get-products \
  --service-code AmazonBedrock \
  --region us-east-1 \
  --output json |
jq -r --arg MODEL "$MODEL_ID" '
  .PriceList[]
  | fromjson
  | select(.product.attributes.modelId? == $MODEL)
  | {
      model: $MODEL,
      type: .product.attributes.inputOutput,
      price_per_1k_tokens: (
        .terms.OnDemand[]
        .priceDimensions[]
        .pricePerUnit.USD
      )
    }
'
