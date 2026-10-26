module tb_nmed;
  logic signed [7:0] a, b;
  logic signed [15:0] z;

  string out_dir, temp_fp;
  int out_file;
  longint count, abs_err, sum_err, max_err, err_count, err, abs_e;

  approx_mult mult (
      .i_a(a),
      .i_b(b),
      .o_z(z)
  );

  initial begin
    if (!$value$plusargs("OUT=%s", out_dir)) $fatal(1, "missing +OUT");
    count = 0;
    abs_err = 0;
    sum_err = 0;
    max_err = 0;
    err_count = 0;

    for (int i = -128; i < 128; i = i + 1) begin
      for (int j = -128; j < 128; j = j + 1) begin
        a = i;
        b = j;
        #1;
        if ($isunknown(z)) $fatal(1, "approx_mult output has X");
        err = longint'(z) - longint'(i * j);
        abs_e = (err < 0) ? -err : err;
        count = count + 1;
        abs_err = abs_err + abs_e;
        sum_err = sum_err + err;
        if (abs_e > max_err) max_err = abs_e;
        if (err != 0) err_count = err_count + 1;
      end
    end

    $sformat(temp_fp, {out_dir, "/nmed_all.txt"});
    out_file = $fopen(temp_fp, "w");
    $fdisplay(out_file, "%0d %0d %0d %0d %0d", count, abs_err, sum_err, max_err, err_count);
    $fclose(out_file);
    $finish;
  end
endmodule : tb_nmed
