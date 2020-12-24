
namespace MachineLearningStudio
{
   partial class PageIntentRetrain
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
         this.textBoxIntent = new System.Windows.Forms.TextBox();
         this.labelDataSetName = new System.Windows.Forms.Label();
         this.labelIntent = new System.Windows.Forms.Label();
         this.textBoxDataSetName = new System.Windows.Forms.TextBox();
         this.textBoxSentence = new System.Windows.Forms.TextBox();
         this.labelSentence = new System.Windows.Forms.Label();
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
         this.tableLayoutPanelMain.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.tableLayoutPanelMain.Name = "tableLayoutPanelMain";
         this.tableLayoutPanelMain.RowCount = 2;
         this.tableLayoutPanelMain.RowStyles.Add(new System.Windows.Forms.RowStyle());
         this.tableLayoutPanelMain.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
         this.tableLayoutPanelMain.Size = new System.Drawing.Size(695, 546);
         this.tableLayoutPanelMain.TabIndex = 0;
         // 
         // panelControls
         // 
         this.panelControls.AutoSize = true;
         this.panelControls.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
         this.panelControls.Controls.Add(this.textBoxIntent);
         this.panelControls.Controls.Add(this.labelDataSetName);
         this.panelControls.Controls.Add(this.labelIntent);
         this.panelControls.Controls.Add(this.textBoxDataSetName);
         this.panelControls.Controls.Add(this.textBoxSentence);
         this.panelControls.Controls.Add(this.labelSentence);
         this.panelControls.Controls.Add(this.buttonTrain);
         this.panelControls.Dock = System.Windows.Forms.DockStyle.Fill;
         this.panelControls.Location = new System.Drawing.Point(4, 3);
         this.panelControls.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.panelControls.Name = "panelControls";
         this.panelControls.Size = new System.Drawing.Size(687, 49);
         this.panelControls.TabIndex = 0;
         // 
         // textBoxIntent
         // 
         this.textBoxIntent.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
         this.textBoxIntent.AutoCompleteMode = System.Windows.Forms.AutoCompleteMode.SuggestAppend;
         this.textBoxIntent.AutoCompleteSource = System.Windows.Forms.AutoCompleteSource.CustomSource;
         this.textBoxIntent.Location = new System.Drawing.Point(534, 23);
         this.textBoxIntent.Name = "textBoxIntent";
         this.textBoxIntent.Size = new System.Drawing.Size(147, 23);
         this.textBoxIntent.TabIndex = 6;
         this.textBoxIntent.KeyDown += new System.Windows.Forms.KeyEventHandler(this.textBoxIntent_KeyDown);
         // 
         // labelDataSetName
         // 
         this.labelDataSetName.AutoSize = true;
         this.labelDataSetName.Location = new System.Drawing.Point(99, 3);
         this.labelDataSetName.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
         this.labelDataSetName.Name = "labelDataSetName";
         this.labelDataSetName.Size = new System.Drawing.Size(49, 15);
         this.labelDataSetName.TabIndex = 1;
         this.labelDataSetName.Text = "Data set";
         // 
         // labelIntent
         // 
         this.labelIntent.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
         this.labelIntent.AutoSize = true;
         this.labelIntent.Location = new System.Drawing.Point(531, 3);
         this.labelIntent.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
         this.labelIntent.Name = "labelIntent";
         this.labelIntent.Size = new System.Drawing.Size(38, 15);
         this.labelIntent.TabIndex = 5;
         this.labelIntent.Text = "Intent";
         // 
         // textBoxDataSetName
         // 
         this.textBoxDataSetName.Location = new System.Drawing.Point(103, 23);
         this.textBoxDataSetName.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.textBoxDataSetName.Name = "textBoxDataSetName";
         this.textBoxDataSetName.Size = new System.Drawing.Size(116, 23);
         this.textBoxDataSetName.TabIndex = 2;
         this.textBoxDataSetName.TextChanged += new System.EventHandler(this.textBoxDataSetName_TextChanged);
         // 
         // textBoxSentence
         // 
         this.textBoxSentence.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
         this.textBoxSentence.Location = new System.Drawing.Point(226, 23);
         this.textBoxSentence.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.textBoxSentence.Name = "textBoxSentence";
         this.textBoxSentence.Size = new System.Drawing.Size(301, 23);
         this.textBoxSentence.TabIndex = 4;
         this.textBoxSentence.TextChanged += new System.EventHandler(this.textBoxSentence_TextChanged);
         // 
         // labelSentence
         // 
         this.labelSentence.AutoSize = true;
         this.labelSentence.Location = new System.Drawing.Point(223, 3);
         this.labelSentence.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
         this.labelSentence.Name = "labelSentence";
         this.labelSentence.Size = new System.Drawing.Size(55, 15);
         this.labelSentence.TabIndex = 3;
         this.labelSentence.Text = "Sentence";
         // 
         // buttonTrain
         // 
         this.buttonTrain.Location = new System.Drawing.Point(4, 3);
         this.buttonTrain.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.buttonTrain.Name = "buttonTrain";
         this.buttonTrain.Size = new System.Drawing.Size(88, 43);
         this.buttonTrain.TabIndex = 0;
         this.buttonTrain.Text = "Train";
         this.buttonTrain.UseVisualStyleBackColor = true;
         this.buttonTrain.Click += new System.EventHandler(this.buttonTrain_Click);
         // 
         // textBoxOutput
         // 
         this.textBoxOutput.Dock = System.Windows.Forms.DockStyle.Fill;
         this.textBoxOutput.Font = new System.Drawing.Font("Consolas", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point);
         this.textBoxOutput.Location = new System.Drawing.Point(4, 58);
         this.textBoxOutput.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.textBoxOutput.MaxLength = 0;
         this.textBoxOutput.Multiline = true;
         this.textBoxOutput.Name = "textBoxOutput";
         this.textBoxOutput.ReadOnly = true;
         this.textBoxOutput.ScrollBars = System.Windows.Forms.ScrollBars.Both;
         this.textBoxOutput.Size = new System.Drawing.Size(687, 485);
         this.textBoxOutput.TabIndex = 1;
         this.textBoxOutput.WordWrap = false;
         // 
         // PageIntentRetrain
         // 
         this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
         this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
         this.Controls.Add(this.tableLayoutPanelMain);
         this.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.Name = "PageIntentRetrain";
         this.Size = new System.Drawing.Size(695, 546);
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
      private System.Windows.Forms.TextBox textBoxIntent;
   }
}
