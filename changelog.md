### v0.3.0 (03/11/2024)
**New Features:**
- Support mock transformers, see [Mock transformers](https://github.com/Oneflow-Inc/libai/tree/main/projects/mock_transformers#readme)
- Support lm-evaluation-harness for model evaluation
- User Experience Optimization

**New Supported Models:**
- These models are natively supported by libai
<table class="docutils">
  <tbody>
    <tr>
      <th width="130"> Models </th>
      <th valign="bottom" align="center" width="140"> 2D(tp+pp) Inference</th>
      <th valign="bottom" align="center" width="140"> 3D Parallel Training </th>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/BLOOM"> <b> BLOOM </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/ChatGLM"> <b> ChatGLM </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">&#10004;</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/Couplets"> <b> Couplets </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">&#10004;</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/DALLE2"> <b> DALLE2 </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/Llama"> <b> Llama2 </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">&#10004;</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/MAE"> <b> MAE </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">&#10004;</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/Oneflow-Inc/libai/tree/main/projects/Stable_Diffusion"> <b> Stable_Diffusion </b> </td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
  </tbody>
</table>

**New Mock Models:**
- These models are extended and implemented by libai through mocking transformers.
<table class="docutils">
  <tbody>
    <tr>
      <th width="130"> Models </th>
      <th valign="bottom" align="center" width="140">Tensor Parallel</th>
      <th valign="bottom" align="center" width="150">Pipeline Parallel</th>
    </tr>
    <tr>
      <td align="center"><a href="https://huggingface.co/docs/transformers/v4.26.1/en/model_doc/bloom#overview"> <b> BLOOM </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/openai/gpt-2/blob/master/model_card.md"> <b> GPT2 </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><a href="https://huggingface.co/docs/transformers/v4.28.0/en/model_doc/llama#overview"> <b> LLAMA </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><a href="https://huggingface.co/docs/transformers/main/en/model_doc/llama2"> <b> LLAMA2 </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><a href="https://huggingface.co/baichuan-inc/Baichuan-7B"> <b> Baichuan </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center"><a href="https://huggingface.co/docs/transformers/v4.26.1/en/model_doc/opt#overview"> <b> OPT </b> </td>
      <td align="center">&#10004;</td>
      <td align="center">-</td>
    </tr>
  </tbody>
</table>