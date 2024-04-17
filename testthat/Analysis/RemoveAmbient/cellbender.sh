cellbender remove-background \
  --input  "~{in_matrix}" \
  --output "~{sample_name}_out.h5" \
  --expected-cells 3000 \
  --total-droplets-included 15000 \
  --fpr  0.01 \
  --model full