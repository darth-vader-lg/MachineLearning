
namespace MachineLearningStudio
{
   partial class PageIntentSdca
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
         this.labelIntent = new System.Windows.Forms.Label();
         this.textBoxDataSetName = new System.Windows.Forms.TextBox();
         this.textBoxSentence = new System.Windows.Forms.TextBox();
         this.labelSentence = new System.Windows.Forms.Label();
         this.buttonTrain = new System.Windows.Forms.Button();
         this.textBoxOutput = new System.Windows.Forms.TextBox();
         this.comboBoxIntent = new System.Windows.Forms.ComboBox();
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
         this.panelControls.Controls.Add(this.comboBoxIntent);
         this.panelControls.Controls.Add(this.labelDataSetName);
         this.panelControls.Controls.Add(this.labelIntent);
         this.panelControls.Controls.Add(this.textBoxDataSetName);
         this.panelControls.Controls.Add(this.textBoxSentence);
         this.panelControls.Controls.Add(this.labelSentence);
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
         // labelIntent
         // 
         this.labelIntent.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
         this.labelIntent.AutoSize = true;
         this.labelIntent.Location = new System.Drawing.Point(456, 3);
         this.labelIntent.Name = "labelIntent";
         this.labelIntent.Size = new System.Drawing.Size(34, 13);
         this.labelIntent.TabIndex = 7;
         this.labelIntent.Text = "Intent";
         // 
         // textBoxDataSetName
         // 
         this.textBoxDataSetName.Location = new System.Drawing.Point(88, 20);
         this.textBoxDataSetName.Name = "textBoxDataSetName";
         this.textBoxDataSetName.Size = new System.Drawing.Size(100, 20);
         this.textBoxDataSetName.TabIndex = 2;
         this.textBoxDataSetName.TextChanged += new System.EventHandler(this.textBoxDataSetName_TextChanged);
         // 
         // textBoxSentence
         // 
         this.textBoxSentence.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
         this.textBoxSentence.Location = new System.Drawing.Point(194, 20);
         this.textBoxSentence.Name = "textBoxSentence";
         this.textBoxSentence.Size = new System.Drawing.Size(259, 20);
         this.textBoxSentence.TabIndex = 4;
         this.textBoxSentence.TextChanged += new System.EventHandler(this.textBoxSentence_TextChanged);
         // 
         // labelSentence
         // 
         this.labelSentence.AutoSize = true;
         this.labelSentence.Location = new System.Drawing.Point(191, 3);
         this.labelSentence.Name = "labelSentence";
         this.labelSentence.Size = new System.Drawing.Size(53, 13);
         this.labelSentence.TabIndex = 3;
         this.labelSentence.Text = "Sentence";
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
         // comboBoxIntent
         // 
         this.comboBoxIntent.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
         this.comboBoxIntent.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
         this.comboBoxIntent.FormattingEnabled = true;
         this.comboBoxIntent.Location = new System.Drawing.Point(459, 19);
         this.comboBoxIntent.Name = "comboBoxIntent";
         this.comboBoxIntent.Size = new System.Drawing.Size(124, 21);
         this.comboBoxIntent.TabIndex = 8;
         this.comboBoxIntent.SelectionChangeCommitted += new System.EventHandler(this.comboBoxIntent_SelectionChangeCommitted);
         // 
         // PageIntentSdca
         // 
         this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
         this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
         this.Controls.Add(this.tableLayoutPanelMain);
         this.Name = "PageIntentSdca";
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
      private System.Windows.Forms.Label labelIntent;
      private System.Windows.Forms.TextBox textBoxSentence;
      private System.Windows.Forms.Label labelSentence;
      private System.Windows.Forms.Label labelDataSetName;
      private System.Windows.Forms.TextBox textBoxDataSetName;
      private System.Windows.Forms.TextBox textBoxOutput;
      private System.Windows.Forms.ComboBox comboBoxIntent;
   }
}
