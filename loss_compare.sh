
# bert loss compare
python projects/bert_loss_compare/train_net.py --config-file projects/bert_loss_compare/configs/compare_loss.py 

# draw loss curve
echo "draw bert loss curve"
python projects/bert_loss_compare/utils/draw_loss_curve.py --torch-loss-path projects/bert_loss_compare/torch_loss.txt

# vit loss compare
python projects/vit_loss_compare/train_net.py --config-file projects/vit_loss_compare/configs/compare_loss.py

# draw loss curve
echo "draw vit loss curve"
python projects/vit_loss_compare/utils/draw_loss_curve.py --torch-loss-path projects/vit_loss_compare/torch_loss.txt
