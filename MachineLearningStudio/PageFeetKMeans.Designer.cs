
namespace TestChoice
{
   partial class PageFeetKMeans
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
         this.panelPlot = new System.Windows.Forms.Panel();
         this.buttonTrain = new System.Windows.Forms.Button();
         this.tableLayoutPanelMain.SuspendLayout();
         this.panelControls.SuspendLayout();
         this.SuspendLayout();
         // 
         // tableLayoutPanelMain
         // 
         this.tableLayoutPanelMain.ColumnCount = 1;
         this.tableLayoutPanelMain.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
         this.tableLayoutPanelMain.Controls.Add(this.panelControls, 0, 0);
         this.tableLayoutPanelMain.Controls.Add(this.panelPlot, 0, 1);
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
         this.panelControls.Controls.Add(this.buttonTrain);
         this.panelControls.Dock = System.Windows.Forms.DockStyle.Fill;
         this.panelControls.Location = new System.Drawing.Point(3, 3);
         this.panelControls.Name = "panelControls";
         this.panelControls.Size = new System.Drawing.Size(590, 29);
         this.panelControls.TabIndex = 0;
         // 
         // panelPlot
         // 
         this.panelPlot.Dock = System.Windows.Forms.DockStyle.Fill;
         this.panelPlot.Location = new System.Drawing.Point(3, 38);
         this.panelPlot.Name = "panelPlot";
         this.panelPlot.Size = new System.Drawing.Size(590, 432);
         this.panelPlot.TabIndex = 1;
         this.panelPlot.Paint += new System.Windows.Forms.PaintEventHandler(this.panelPlot_Paint);
         this.panelPlot.Resize += new System.EventHandler(this.panelPlot_Resize);
         // 
         // buttonTrain
         // 
         this.buttonTrain.Location = new System.Drawing.Point(3, 3);
         this.buttonTrain.Name = "buttonTrain";
         this.buttonTrain.Size = new System.Drawing.Size(75, 23);
         this.buttonTrain.TabIndex = 2;
         this.buttonTrain.Text = "Train";
         this.buttonTrain.UseVisualStyleBackColor = true;
         this.buttonTrain.Click += new System.EventHandler(this.buttonTrain_Click);
         // 
         // PageFeetKMeans
         // 
         this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
         this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
         this.Controls.Add(this.tableLayoutPanelMain);
         this.Name = "PageFeetKMeans";
         this.Size = new System.Drawing.Size(596, 473);
         this.tableLayoutPanelMain.ResumeLayout(false);
         this.tableLayoutPanelMain.PerformLayout();
         this.panelControls.ResumeLayout(false);
         this.ResumeLayout(false);

      }

      #endregion

      private System.Windows.Forms.TableLayoutPanel tableLayoutPanelMain;
      private System.Windows.Forms.Panel panelControls;
      private System.Windows.Forms.Panel panelPlot;
      private System.Windows.Forms.Button buttonTrain;
   }
}
