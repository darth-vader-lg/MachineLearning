
namespace MachineLearningStudio
{
   partial class PageImageClassification
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
         this.labelImageSetName = new System.Windows.Forms.Label();
         this.labelClassResult = new System.Windows.Forms.Label();
         this.labelClass = new System.Windows.Forms.Label();
         this.textBoxImageSetName = new System.Windows.Forms.TextBox();
         this.buttonLoad = new System.Windows.Forms.Button();
         this.splitContainerImageAndMetrics = new System.Windows.Forms.SplitContainer();
         this.pictureBox = new System.Windows.Forms.PictureBox();
         this.textBoxOutput = new System.Windows.Forms.TextBox();
         this.openFileDialog = new System.Windows.Forms.OpenFileDialog();
         this.checkBoxCrossValidate = new System.Windows.Forms.CheckBox();
         this.tableLayoutPanelMain.SuspendLayout();
         this.panelControls.SuspendLayout();
         ((System.ComponentModel.ISupportInitialize)(this.splitContainerImageAndMetrics)).BeginInit();
         this.splitContainerImageAndMetrics.Panel1.SuspendLayout();
         this.splitContainerImageAndMetrics.Panel2.SuspendLayout();
         this.splitContainerImageAndMetrics.SuspendLayout();
         ((System.ComponentModel.ISupportInitialize)(this.pictureBox)).BeginInit();
         this.SuspendLayout();
         // 
         // tableLayoutPanelMain
         // 
         this.tableLayoutPanelMain.ColumnCount = 1;
         this.tableLayoutPanelMain.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
         this.tableLayoutPanelMain.Controls.Add(this.panelControls, 0, 0);
         this.tableLayoutPanelMain.Controls.Add(this.splitContainerImageAndMetrics, 0, 1);
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
         this.panelControls.Controls.Add(this.checkBoxCrossValidate);
         this.panelControls.Controls.Add(this.labelImageSetName);
         this.panelControls.Controls.Add(this.labelClassResult);
         this.panelControls.Controls.Add(this.labelClass);
         this.panelControls.Controls.Add(this.textBoxImageSetName);
         this.panelControls.Controls.Add(this.buttonLoad);
         this.panelControls.Dock = System.Windows.Forms.DockStyle.Fill;
         this.panelControls.Location = new System.Drawing.Point(3, 3);
         this.panelControls.Name = "panelControls";
         this.panelControls.Size = new System.Drawing.Size(590, 43);
         this.panelControls.TabIndex = 0;
         // 
         // labelImageSetName
         // 
         this.labelImageSetName.AutoSize = true;
         this.labelImageSetName.Location = new System.Drawing.Point(180, 2);
         this.labelImageSetName.Name = "labelImageSetName";
         this.labelImageSetName.Size = new System.Drawing.Size(53, 13);
         this.labelImageSetName.TabIndex = 1;
         this.labelImageSetName.Text = "Image set";
         // 
         // labelClassResult
         // 
         this.labelClassResult.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
         this.labelClassResult.Location = new System.Drawing.Point(289, 19);
         this.labelClassResult.Name = "labelClassResult";
         this.labelClassResult.Size = new System.Drawing.Size(100, 20);
         this.labelClassResult.TabIndex = 8;
         this.labelClassResult.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
         // 
         // labelClass
         // 
         this.labelClass.AutoSize = true;
         this.labelClass.Location = new System.Drawing.Point(286, 2);
         this.labelClass.Name = "labelClass";
         this.labelClass.Size = new System.Drawing.Size(32, 13);
         this.labelClass.TabIndex = 7;
         this.labelClass.Text = "Class";
         // 
         // textBoxImageSetName
         // 
         this.textBoxImageSetName.Location = new System.Drawing.Point(183, 19);
         this.textBoxImageSetName.Name = "textBoxImageSetName";
         this.textBoxImageSetName.Size = new System.Drawing.Size(100, 20);
         this.textBoxImageSetName.TabIndex = 2;
         this.textBoxImageSetName.TextChanged += new System.EventHandler(this.textBoxImageSetName_TextChanged);
         // 
         // buttonLoad
         // 
         this.buttonLoad.Location = new System.Drawing.Point(3, 3);
         this.buttonLoad.Name = "buttonLoad";
         this.buttonLoad.Size = new System.Drawing.Size(75, 37);
         this.buttonLoad.TabIndex = 0;
         this.buttonLoad.Text = "Load";
         this.buttonLoad.UseVisualStyleBackColor = true;
         this.buttonLoad.Click += new System.EventHandler(this.buttonLoad_Click);
         // 
         // splitContainerImageAndMetrics
         // 
         this.splitContainerImageAndMetrics.Dock = System.Windows.Forms.DockStyle.Fill;
         this.splitContainerImageAndMetrics.Location = new System.Drawing.Point(3, 52);
         this.splitContainerImageAndMetrics.Name = "splitContainerImageAndMetrics";
         this.splitContainerImageAndMetrics.Orientation = System.Windows.Forms.Orientation.Horizontal;
         // 
         // splitContainerImageAndMetrics.Panel1
         // 
         this.splitContainerImageAndMetrics.Panel1.Controls.Add(this.pictureBox);
         // 
         // splitContainerImageAndMetrics.Panel2
         // 
         this.splitContainerImageAndMetrics.Panel2.Controls.Add(this.textBoxOutput);
         this.splitContainerImageAndMetrics.Size = new System.Drawing.Size(590, 418);
         this.splitContainerImageAndMetrics.SplitterDistance = 312;
         this.splitContainerImageAndMetrics.TabIndex = 1;
         // 
         // pictureBox
         // 
         this.pictureBox.Dock = System.Windows.Forms.DockStyle.Fill;
         this.pictureBox.Location = new System.Drawing.Point(0, 0);
         this.pictureBox.Name = "pictureBox";
         this.pictureBox.Size = new System.Drawing.Size(590, 312);
         this.pictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
         this.pictureBox.TabIndex = 1;
         this.pictureBox.TabStop = false;
         // 
         // textBoxOutput
         // 
         this.textBoxOutput.Dock = System.Windows.Forms.DockStyle.Fill;
         this.textBoxOutput.Location = new System.Drawing.Point(0, 0);
         this.textBoxOutput.MaxLength = 0;
         this.textBoxOutput.Multiline = true;
         this.textBoxOutput.Name = "textBoxOutput";
         this.textBoxOutput.ReadOnly = true;
         this.textBoxOutput.Size = new System.Drawing.Size(590, 102);
         this.textBoxOutput.TabIndex = 0;
         // 
         // openFileDialog
         // 
         this.openFileDialog.Filter = "Image files|*.jpg;*.png;*.bmp";
         // 
         // checkBoxCrossValidate
         // 
         this.checkBoxCrossValidate.AutoSize = true;
         this.checkBoxCrossValidate.Location = new System.Drawing.Point(85, 22);
         this.checkBoxCrossValidate.Name = "checkBoxCrossValidate";
         this.checkBoxCrossValidate.Size = new System.Drawing.Size(92, 17);
         this.checkBoxCrossValidate.TabIndex = 9;
         this.checkBoxCrossValidate.Text = "Cross validate";
         this.checkBoxCrossValidate.UseVisualStyleBackColor = true;
         this.checkBoxCrossValidate.CheckedChanged += new System.EventHandler(this.checkBoxCrossValidate_CheckedChanged);
         // 
         // PageImageClassification
         // 
         this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
         this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
         this.Controls.Add(this.tableLayoutPanelMain);
         this.Name = "PageImageClassification";
         this.Size = new System.Drawing.Size(596, 473);
         this.tableLayoutPanelMain.ResumeLayout(false);
         this.tableLayoutPanelMain.PerformLayout();
         this.panelControls.ResumeLayout(false);
         this.panelControls.PerformLayout();
         this.splitContainerImageAndMetrics.Panel1.ResumeLayout(false);
         this.splitContainerImageAndMetrics.Panel2.ResumeLayout(false);
         this.splitContainerImageAndMetrics.Panel2.PerformLayout();
         ((System.ComponentModel.ISupportInitialize)(this.splitContainerImageAndMetrics)).EndInit();
         this.splitContainerImageAndMetrics.ResumeLayout(false);
         ((System.ComponentModel.ISupportInitialize)(this.pictureBox)).EndInit();
         this.ResumeLayout(false);

      }

      #endregion

      private System.Windows.Forms.TableLayoutPanel tableLayoutPanelMain;
      private System.Windows.Forms.Panel panelControls;
      private System.Windows.Forms.Button buttonLoad;
      private System.Windows.Forms.Label labelClassResult;
      private System.Windows.Forms.Label labelClass;
      private System.Windows.Forms.Label labelImageSetName;
      private System.Windows.Forms.TextBox textBoxImageSetName;
      private System.Windows.Forms.PictureBox pictureBox;
      private System.Windows.Forms.SplitContainer splitContainerImageAndMetrics;
      private System.Windows.Forms.TextBox textBoxOutput;
      private System.Windows.Forms.OpenFileDialog openFileDialog;
      private System.Windows.Forms.CheckBox checkBoxCrossValidate;
   }
}
