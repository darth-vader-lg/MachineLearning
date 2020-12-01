
namespace MachineLearningStudio
{
   partial class PageFeetSdca
   {
      /// <summary> 
      /// Variabile di progettazione necessaria.
      /// </summary>
      private System.ComponentModel.IContainer components = null;

      /// <summary> 
      /// Pulire le risorse in uso.
      /// </summary>
      /// <param name="disposing">ha valore true se le risorse gestite devono essere eliminate, false in caso contrario.</param>
      protected override void Dispose(bool disposing)
      {
         if (disposing && (components != null)) {
            components.Dispose();
         }
         base.Dispose(disposing);
      }

      #region Codice generato da Progettazione componenti

      /// <summary> 
      /// Metodo necessario per il supporto della finestra di progettazione. Non modificare 
      /// il contenuto del metodo con l'editor di codice.
      /// </summary>
      private void InitializeComponent()
      {
         this.tableLayoutPanelMain = new System.Windows.Forms.TableLayoutPanel();
         this.panelControls = new System.Windows.Forms.Panel();
         this.labelDataSetName = new System.Windows.Forms.Label();
         this.labelNumberResult = new System.Windows.Forms.Label();
         this.textBoxInstep = new System.Windows.Forms.TextBox();
         this.labelNumber = new System.Windows.Forms.Label();
         this.labelInstep = new System.Windows.Forms.Label();
         this.textBoxDataSetName = new System.Windows.Forms.TextBox();
         this.textBoxLength = new System.Windows.Forms.TextBox();
         this.labelLength = new System.Windows.Forms.Label();
         this.buttonTrain = new System.Windows.Forms.Button();
         this.textBoxOutput = new System.Windows.Forms.TextBox();
         this.tableLayoutPanelMain.SuspendLayout();
         this.panelControls.SuspendLayout();
         this.SuspendLayout();
         // 
         // tableLayoutPanelMain
         // 
         this.tableLayoutPanelMain.ColumnCount = 1;
         this.tableLayoutPanelMain.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
         this.tableLayoutPanelMain.Controls.Add(this.panelControls, 0, 0);
         this.tableLayoutPanelMain.Controls.Add(this.textBoxOutput, 0, 1);
         this.tableLayoutPanelMain.Dock = System.Windows.Forms.DockStyle.Fill;
         this.tableLayoutPanelMain.Location = new System.Drawing.Point(0, 0);
         this.tableLayoutPanelMain.Name = "tableLayoutPanelMain";
         this.tableLayoutPanelMain.RowCount = 2;
         this.tableLayoutPanelMain.RowStyles.Add(new System.Windows.Forms.RowStyle());
         this.tableLayoutPanelMain.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
         this.tableLayoutPanelMain.Size = new System.Drawing.Size(596, 473);
         this.tableLayoutPanelMain.TabIndex = 0;
         // 
         // panelControls
         // 
         this.panelControls.AutoSize = true;
         this.panelControls.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
         this.panelControls.Controls.Add(this.labelDataSetName);
         this.panelControls.Controls.Add(this.labelNumberResult);
         this.panelControls.Controls.Add(this.textBoxInstep);
         this.panelControls.Controls.Add(this.labelNumber);
         this.panelControls.Controls.Add(this.labelInstep);
         this.panelControls.Controls.Add(this.textBoxDataSetName);
         this.panelControls.Controls.Add(this.textBoxLength);
         this.panelControls.Controls.Add(this.labelLength);
         this.panelControls.Controls.Add(this.buttonTrain);
         this.panelControls.Dock = System.Windows.Forms.DockStyle.Fill;
         this.panelControls.Location = new System.Drawing.Point(3, 3);
         this.panelControls.Name = "panelControls";
         this.panelControls.Size = new System.Drawing.Size(590, 43);
         this.panelControls.TabIndex = 0;
         // 
         // labelDataSetName
         // 
         this.labelDataSetName.AutoSize = true;
         this.labelDataSetName.Location = new System.Drawing.Point(85, 3);
         this.labelDataSetName.Name = "labelDataSetName";
         this.labelDataSetName.Size = new System.Drawing.Size(47, 13);
         this.labelDataSetName.TabIndex = 1;
         this.labelDataSetName.Text = "Data set";
         // 
         // labelNumberResult
         // 
         this.labelNumberResult.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
         this.labelNumberResult.Location = new System.Drawing.Point(406, 20);
         this.labelNumberResult.Name = "labelNumberResult";
         this.labelNumberResult.Size = new System.Drawing.Size(45, 20);
         this.labelNumberResult.TabIndex = 8;
         this.labelNumberResult.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
         // 
         // textBoxInstep
         // 
         this.textBoxInstep.Location = new System.Drawing.Point(300, 20);
         this.textBoxInstep.Name = "textBoxInstep";
         this.textBoxInstep.Size = new System.Drawing.Size(100, 20);
         this.textBoxInstep.TabIndex = 6;
         this.textBoxInstep.TextChanged += new System.EventHandler(this.textBoxInstep_TextChanged);
         // 
         // labelNumber
         // 
         this.labelNumber.AutoSize = true;
         this.labelNumber.Location = new System.Drawing.Point(403, 3);
         this.labelNumber.Name = "labelNumber";
         this.labelNumber.Size = new System.Drawing.Size(44, 13);
         this.labelNumber.TabIndex = 7;
         this.labelNumber.Text = "Number";
         // 
         // labelInstep
         // 
         this.labelInstep.AutoSize = true;
         this.labelInstep.Location = new System.Drawing.Point(297, 3);
         this.labelInstep.Name = "labelInstep";
         this.labelInstep.Size = new System.Drawing.Size(36, 13);
         this.labelInstep.TabIndex = 5;
         this.labelInstep.Text = "Instep";
         // 
         // textBoxDataSetName
         // 
         this.textBoxDataSetName.Location = new System.Drawing.Point(88, 20);
         this.textBoxDataSetName.Name = "textBoxDataSetName";
         this.textBoxDataSetName.Size = new System.Drawing.Size(100, 20);
         this.textBoxDataSetName.TabIndex = 2;
         this.textBoxDataSetName.TextChanged += new System.EventHandler(this.textBoxDataSetName_TextChanged);
         // 
         // textBoxLength
         // 
         this.textBoxLength.Location = new System.Drawing.Point(194, 20);
         this.textBoxLength.Name = "textBoxLength";
         this.textBoxLength.Size = new System.Drawing.Size(100, 20);
         this.textBoxLength.TabIndex = 4;
         this.textBoxLength.TextChanged += new System.EventHandler(this.textBoxLength_TextChanged);
         // 
         // labelLength
         // 
         this.labelLength.AutoSize = true;
         this.labelLength.Location = new System.Drawing.Point(191, 3);
         this.labelLength.Name = "labelLength";
         this.labelLength.Size = new System.Drawing.Size(40, 13);
         this.labelLength.TabIndex = 3;
         this.labelLength.Text = "Length";
         // 
         // buttonTrain
         // 
         this.buttonTrain.Location = new System.Drawing.Point(3, 3);
         this.buttonTrain.Name = "buttonTrain";
         this.buttonTrain.Size = new System.Drawing.Size(75, 37);
         this.buttonTrain.TabIndex = 0;
         this.buttonTrain.Text = "Train";
         this.buttonTrain.UseVisualStyleBackColor = true;
         this.buttonTrain.Click += new System.EventHandler(this.buttonTrain_Click);
         // 
         // textBoxOutput
         // 
         this.textBoxOutput.Dock = System.Windows.Forms.DockStyle.Fill;
         this.textBoxOutput.Location = new System.Drawing.Point(3, 52);
         this.textBoxOutput.MaxLength = 0;
         this.textBoxOutput.Multiline = true;
         this.textBoxOutput.Name = "textBoxOutput";
         this.textBoxOutput.ReadOnly = true;
         this.textBoxOutput.Size = new System.Drawing.Size(590, 418);
         this.textBoxOutput.TabIndex = 1;
         // 
         // PageFeetSdca
         // 
         this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
         this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
         this.Controls.Add(this.tableLayoutPanelMain);
         this.Name = "PageFeetSdca";
         this.Size = new System.Drawing.Size(596, 473);
         this.tableLayoutPanelMain.ResumeLayout(false);
         this.tableLayoutPanelMain.PerformLayout();
         this.panelControls.ResumeLayout(false);
         this.panelControls.PerformLayout();
         this.ResumeLayout(false);

      }

      #endregion

      private System.Windows.Forms.TableLayoutPanel tableLayoutPanelMain;
      private System.Windows.Forms.Panel panelControls;
      private System.Windows.Forms.Button buttonTrain;
      private System.Windows.Forms.Label labelNumberResult;
      private System.Windows.Forms.TextBox textBoxInstep;
      private System.Windows.Forms.Label labelNumber;
      private System.Windows.Forms.Label labelInstep;
      private System.Windows.Forms.TextBox textBoxLength;
      private System.Windows.Forms.Label labelLength;
      private System.Windows.Forms.Label labelDataSetName;
      private System.Windows.Forms.TextBox textBoxDataSetName;
      private System.Windows.Forms.TextBox textBoxOutput;
   }
}
