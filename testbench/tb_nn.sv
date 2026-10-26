function int findCommaInString(string in_string);
  findCommaInString = -1;
  for (int idx = 0; idx < in_string.len(); idx = idx + 1) begin
    if (in_string[idx] == ",") begin
      findCommaInString = idx;
      break;
    end
  end
endfunction : findCommaInString

module tb_nn;
  int layer_count = 3;

  logic [7:0] i_mulA1, i_mulA2, i_mulB1, i_mulB2;
  logic [15:0] o_mulP1, o_mulP2;

  string model_dir, exact_dir, inputs_dir, layers_dir, outputs_dir, temp_fp, line;

  int temp;
  int zp_file;
  int comma_loc;
  int meta_file;
  int layer_type;
  int quant_file;
  int layer_start;
  int layer_bdims;
  int output_file;
  int layer_idims[];
  int layer_wdims[];
  int layer_odims[];
  int input_zp_dims;
  int output_zp_dims;
  int weight_zp_dims;
  int last_layer_mode;
  int output_quant_mult_dims;
  int output_quant_shift_dims;

  int operation_count;
  int input_count;
  int saturate;

  longint layer_n[3], layer_abs[3], layer_err[3], layer_maxex[3], layer_maxerr[3], layer_nerr[3];
  longint exact_p, err_p, sat64;
  int stats_file;

  logic signed [31:0] layer_b[];
  logic signed [7:0] layer_w[];
  logic signed [7:0] layer_i[];
  logic signed [7:0] layer_o[];

  logic signed [15:0] shifted_input;
  logic signed [7:0] shifted_filter;
  logic signed [7:0] in8A;
  logic signed [7:0] in8B;

  logic signed [7:0] input_zp;
  logic signed [7:0] output_zp;
  logic signed [7:0] filter_zp;

  logic signed [31:0] quant_mult;
  logic signed [31:0] quant_shift;
  logic signed [31:0] total_shift;

  logic signed [15:0] product16;
  logic signed [15:0] offset16;
  logic signed [31:0] product32;
  logic signed [31:0] relu32;
  logic signed [63:0] product64;
  logic signed [63:0] shifted_product64;
  logic signed [63:0] round;
  logic signed [31:0] acc32;

  mult16bvia8bit mult (
      .i_a(shifted_input),
      .i_b(offset16),
      .o_z(product32)
  );

  initial begin
    if (!$value$plusargs("MODEL=%s", model_dir)) $fatal(1, "missing +MODEL");
    if (!$value$plusargs("OUT=%s", outputs_dir)) $fatal(1, "missing +OUT");
    if (!$value$plusargs("N=%d", input_count)) input_count = 100;
    if (!$value$plusargs("SAT=%d", saturate)) saturate = 0;
    if (!$value$plusargs("LAST=%d", last_layer_mode)) last_layer_mode = 0;
    if (!$value$plusargs("EXACT=%s", exact_dir)) exact_dir = outputs_dir;

    inputs_dir = {model_dir, "/inputs/"};
    layers_dir = {model_dir, "/layers/"};
    outputs_dir = {outputs_dir, "/"};
    exact_dir = {exact_dir, "/"};
    for (int k = 0; k < 3; k++) begin
      layer_n[k] = 0;
      layer_abs[k] = 0;
      layer_err[k] = 0;
      layer_maxex[k] = 0;
      layer_maxerr[k] = 0;
      layer_nerr[k] = 0;
    end
    layer_idims = '{0, 0};
    layer_wdims = '{0, 0};
    layer_odims = '{0, 0};

    layer_start = 0;
    if (last_layer_mode) begin
      layer_start = 2;
    end


    // network ops
    for (int input_idx = 0; input_idx < input_count; input_idx = input_idx + 1) begin
      for (int layer_id = layer_start; layer_id < layer_count; layer_id = layer_id + 1) begin
        // load layer details from metadata
        $sformat(temp_fp, {layers_dir, "%0d_meta.txt"}, layer_id);
        meta_file = $fopen(temp_fp, "r");

        temp = $fscanf(meta_file, "%s", line);
        // $display("ID: %s", line);

        temp = $fscanf(meta_file, "%s", line);
        layer_type = line.atoi();
        // $display("Type: %0d", layer_type);

        temp = $fscanf(meta_file, "%s", line);
        comma_loc = findCommaInString(line);
        layer_idims[0] = line.substr(0, comma_loc).atoi();
        layer_idims[1] = line.substr(comma_loc + 1, line.len() - 1).atoi();
        // $display("Input dims: {%0d, %0d}", layer_idims[0], layer_idims[1]);

        temp = $fscanf(meta_file, "%s", line);
        comma_loc = findCommaInString(line);
        layer_odims[0] = line.substr(0, comma_loc).atoi();
        layer_odims[1] = line.substr(comma_loc + 1, line.len() - 1).atoi();
        // $display("Output dims: {%0d, %0d}", layer_odims[0], layer_odims[1]);

        temp = $fscanf(meta_file, "%s", line);
        comma_loc = findCommaInString(line);
        layer_wdims[0] = line.substr(0, comma_loc).atoi();
        layer_wdims[1] = line.substr(comma_loc + 1, line.len() - 1).atoi();
        // $display("Weight dims: {%0d, %0d}", layer_wdims[0], layer_wdims[1]);

        temp = $fscanf(meta_file, "%s", line);
        layer_bdims = line.atoi();
        // $display("Bias dims: %0d", layer_bdims);

        temp = $fscanf(meta_file, "%s", line);
        input_zp_dims = line.atoi();

        temp = $fscanf(meta_file, "%s", line);
        output_zp_dims = line.atoi();

        temp = $fscanf(meta_file, "%s", line);
        weight_zp_dims = line.atoi();

        temp = $fscanf(meta_file, "%s", line);
        output_quant_mult_dims = line.atoi();

        temp = $fscanf(meta_file, "%s", line);
        output_quant_shift_dims = line.atoi();

        $fclose(meta_file);

        // initialize layer parameters
        layer_i = new[layer_idims[0] * layer_idims[1]];
        layer_o = new[layer_odims[0] * layer_odims[1]];
        layer_w = new[layer_wdims[0] * layer_wdims[1]];
        layer_b = new[layer_bdims];

        $sformat(temp_fp, {layers_dir, "%0d_zeropoints.txt"}, layer_id);
        zp_file = $fopen(temp_fp, "r");
        temp = $fscanf(zp_file, "%b", input_zp);
        temp = $fscanf(zp_file, "%b", output_zp);
        temp = $fscanf(zp_file, "%b", filter_zp);
        $fclose(zp_file);

        $sformat(temp_fp, {layers_dir, "%0d_quants.txt"}, layer_id);
        // $display("%s", temp_fp);
        quant_file = $fopen(temp_fp, "r");
        temp = $fscanf(quant_file, "%b", quant_mult);
        temp = $fscanf(quant_file, "%b", quant_shift);
        // $display("%d, %d", quant_mult, quant_shift);
        $fclose(quant_file);


        // load layer data
        $sformat(temp_fp, {layers_dir, "%0d_filters.txt"}, layer_id);
        $readmemb(temp_fp, layer_w);
        $sformat(temp_fp, {layers_dir, "%0d_bias.txt"}, layer_id);
        $readmemb(temp_fp, layer_b);

        // load input data
        if (layer_id && !last_layer_mode) begin
          $sformat(temp_fp, {outputs_dir, "%0d_layer%0d.txt"}, input_idx, layer_id - 1);
          $readmemb(temp_fp, layer_i);
        end else if (layer_id && last_layer_mode) begin
          $sformat(temp_fp, {exact_dir, "%0d_layer%0d.txt"}, input_idx, layer_id - 1);
          $readmemb(temp_fp, layer_i);
        end else begin
          $sformat(temp_fp, {inputs_dir, "%0d_bin_inmnist.txt"}, input_idx);
          $readmemb(temp_fp, layer_i);
        end

        // Perform dot product (weights are transposed dim0->dim1, dim1->dim0)
        operation_count = 0;
        for (int dim1 = 0; dim1 < layer_wdims[0]; dim1 = dim1 + 1) begin
          acc32 = 0;
          for (int dim0 = 0; dim0 < layer_wdims[1]; dim0 = dim0 + 1) begin
            shifted_input = layer_i[dim0] - input_zp;
            offset16 = layer_w[dim1*layer_wdims[1]+dim0] - filter_zp;
            #10;
            if ($isunknown(product32)) $fatal(1, "approx_mult output has X");
            exact_p = longint'(shifted_input) * longint'(offset16);
            err_p = longint'(product32) - exact_p;
            layer_n[layer_id] = layer_n[layer_id] + 1;
            layer_abs[layer_id] = layer_abs[layer_id] + ((err_p < 0) ? -err_p : err_p);
            layer_err[layer_id] = layer_err[layer_id] + err_p;
            if (err_p != 0) layer_nerr[layer_id] = layer_nerr[layer_id] + 1;
            if (((exact_p < 0) ? -exact_p : exact_p) > layer_maxex[layer_id])
              layer_maxex[layer_id] = (exact_p < 0) ? -exact_p : exact_p;
            if (((err_p < 0) ? -err_p : err_p) > layer_maxerr[layer_id])
              layer_maxerr[layer_id] = (err_p < 0) ? -err_p : err_p;
            acc32 = acc32 + product32;
          end
          acc32 = acc32 + layer_b[dim1];
          if (layer_type) relu32 = acc32;
          else relu32 = acc32[31] ? 0 : acc32;
          //   $display("%d", product32);
          total_shift = 31 - quant_shift;
          round = 1 << (total_shift - 1);
          //   $display("%d", round);
          product64 = relu32 * quant_mult;
          product64 = product64 + round;
          shifted_product64 = product64 >>> total_shift;
          sat64 = shifted_product64 + output_zp;
          if (saturate && sat64 > 127) layer_o[dim1] = 127;
          else if (saturate && sat64 < -128) layer_o[dim1] = -128;
          else layer_o[dim1] = sat64[7:0];
        end
        $sformat(temp_fp, {outputs_dir, "%0d_layer%0d.txt"}, input_idx, layer_id);
        output_file = $fopen(temp_fp, "w");
        for (int i = 0; i < layer_o.size; i = i + 1) begin
          $fdisplay(output_file, "%b", layer_o[i]);
        end
        $fclose(output_file);

        layer_i.delete();
        layer_w.delete();
        layer_o.delete();
        layer_b.delete();
      end
      if (input_idx % 10 == 0 && input_idx != 0)
        $display("Completed %0d inputs out of %0d", input_idx, input_count);
    end
    $display("Completed %0d inputs out of %0d", input_count, input_count);

    $sformat(temp_fp, {outputs_dir, "stats.txt"});
    stats_file = $fopen(temp_fp, "w");
    for (int k = 0; k < 3; k = k + 1)
      $fdisplay(stats_file, "%0d %0d %0d %0d %0d %0d %0d", k, layer_n[k], layer_abs[k],
                layer_err[k], layer_maxex[k], layer_maxerr[k], layer_nerr[k]);
    $fclose(stats_file);
    $finish;
  end

endmodule : tb_nn
