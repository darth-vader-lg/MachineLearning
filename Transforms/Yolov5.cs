﻿using MachineLearning.Data;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Globalization;
using System.Linq;

namespace MachineLearning.Transforms
{
   /// <summary>
   /// Estimator dei dati di uscita Yolov5 in boxes, scores e classes
   /// </summary>
   public sealed class Yolov5Estimator : IEstimator<Yolov5Transformer>
   {
      #region Properties
      /// <summary>
      /// Catalogo trasformazioni
      /// </summary>
      private TransformsCatalog Transforms { get; }
      /// <summary>
      /// Opzioni
      /// </summary>
      private Yolov5Transformer.Options Options { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="transformsCatalog">Catalogo di trasformazioni</param>
      /// <param name="options">Opzioni</param>
      internal Yolov5Estimator(TransformsCatalog transformsCatalog, Yolov5Transformer.Options options)
      {
         Transforms = transformsCatalog;
         Options = options;
      }
      /// <summary>
      /// Ottiene il transformer
      /// </summary>
      /// <param name="input">Dati di input</param>
      /// <returns>Il transformer</returns>
      public Yolov5Transformer Fit(IDataView input) => new(Transforms, Options, input.Schema);
      /// <summary>
      /// Restituisce lo schema di output dell'estimatore
      /// </summary>
      /// <param name="inputSchema">Lo schema di input</param>
      /// <returns>Lo schema di output</returns>
      public SchemaShape GetOutputSchema(SchemaShape inputSchema) => new Yolov5Transformer(Transforms, Options).Pipe.GetOutputSchema(inputSchema);
      #endregion
   }

   /// <summary>
   /// Transformer
   /// </summary>
   public sealed partial class Yolov5Transformer : TransformerBase
   {
      #region Properties
      /// <summary>
      /// Schema di input
      /// </summary>
      public sealed override DataViewSchema InputSchema { get; }
      /// <summary>
      /// Pipe di trasformazione
      /// </summary>
      internal sealed override IEstimator<ITransformer> Pipe { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="transformsCatalog">Catalogo di trasformazioni</param>
      /// <param name="options">Opzioni</param>
      /// <param name="inputSchema">Schema di input</param>
      internal Yolov5Transformer(TransformsCatalog transformsCatalog, Options options, DataViewSchema inputSchema = null) : base(transformsCatalog)
      {
         // Lista di estimatori
         var estimators = new EstimatorList();
         // Rinomina la colonna di input se ha un nome diverso dallo standard
         if (options.InputColumnName != nameof(Mapper.Input.detection))
            estimators.Add(transformsCatalog.CopyColumns(nameof(Mapper.Input.detection), options.InputColumnName));
         // Aggiunge le informazioni sul modello in modo che possano essere sia salvate che rilette dal custom mapper
         estimators.Add(
            transformsCatalog.AddConst(
               nameof(Mapper.Input.D73BD0CB_FEA4_4EC9_9CD0_74B88DFF44F2),
               FormattableString.Invariant($"\"{options.ModelImageWidth}|{options.ModelImageHeight}|{options.MinScoreConfidence}|{options.MinPerCategoryConfidence}|{options.NmsOverlapRatio}\"")));
         // Aggiunge la mappatura custom per la trasformazione dell'output Yolov5 in output standard dei modelli di rilevamento oggetti
         estimators.Add(transformsCatalog.CustomMapping(new Mapper().GetMapping(), nameof(Mapper.Input.D73BD0CB_FEA4_4EC9_9CD0_74B88DFF44F2)));
         // Colonne di output
         var outputColumnNames = new[] {
            new { Standard = nameof(Mapper.Output.detection_boxes), Requested = options.BoxesColumnName },
            new { Standard = nameof(Mapper.Output.detection_classes), Requested = options.ClassesColumnName },
            new { Standard = nameof(Mapper.Output.detection_scores), Requested = options.ScoresColumnName }
         };
         // Rinomina le colonne di output se hanno nomi diversi da quelli standard
         foreach (var names in outputColumnNames) {
            if (names.Standard != names.Requested)
               estimators.Add(transformsCatalog.CopyColumns(names.Requested, names.Standard));
         }
         // Elimina le colonne temporanee diverse da quelle specificate dai parametri
         var dropColumns =
            outputColumnNames
            .Where(names => names.Standard != names.Requested)
            .Select(names => names.Standard)
            .Concat(new[] { nameof(Mapper.Input.detection) }
            .Where(c => c != options.InputColumnName))
            .Concat(new[] { nameof(Mapper.Input.D73BD0CB_FEA4_4EC9_9CD0_74B88DFF44F2) })
            .ToArray();
         if (dropColumns.Length > 0)
            estimators.Add(transformsCatalog.DropColumns(dropColumns));
         Pipe = estimators.GetPipe();
         InputSchema = inputSchema ?? DataViewSchemaBuilder.Build((Name: options.InputColumnName, Type: typeof(float[])));
      }
      #endregion
   }

   /// <summary>
   /// Custom mapper di trasformazione dei dati
   /// </summary>
   public sealed partial class Yolov5Transformer // Mapper
   {
      [CustomMappingFactoryAttribute(nameof(Input.D73BD0CB_FEA4_4EC9_9CD0_74B88DFF44F2))]
      private class Mapper : CustomMappingFactory<Mapper.Input, Mapper.Output>
      {
         #region Input
         /// <summary>
         /// Dati di input da modello Yolo
         /// </summary>
         public class Input
         {
            #region Properties
            /// <summary>
            /// Nome colonna contenente le informazioni sul modello
            /// </summary>
            public string D73BD0CB_FEA4_4EC9_9CD0_74B88DFF44F2 { get; set; }
            /// <summary>
            /// Array di dati di rilevamento
            /// </summary>
            [SuppressMessage("Style", "IDE1006:Stili di denominazione", Justification = "Deve corrispondere al nome in uscita dal modello")]
            public float[] detection { get; set; }
            #endregion
         };
         #endregion
         #region Output
         /// <summary>
         /// Dati di output del transformer
         /// </summary>
         public class Output
         {
            #region Properties
            /// <summary>
            /// Array di classi rilevate
            /// </summary>
            [SuppressMessage("Style", "IDE1006:Stili di denominazione", Justification = "Deve corrispondere al nome in uscita dal modello")]
            public float[] detection_classes { get; set; }
            /// <summary>
            /// Array di punteggi di rilevamento
            /// </summary>
            [SuppressMessage("Style", "IDE1006:Stili di denominazione", Justification = "Deve corrispondere al nome in uscita dal modello")]
            public float[] detection_scores { get; set; }
            /// <summary>
            /// Array di riquadri di rilevamento in quartetti left, top, right, bottom da 0 a 1
            /// </summary>
            [SuppressMessage("Style", "IDE1006:Stili di denominazione", Justification = "Deve corrispondere al nome in uscita dal modello")]
            public float[] detection_boxes { get; set; }
            #endregion
         };
         #endregion
         #region Methods
         /// <summary>
         /// Restituisce l'azzione di mappatura personalizzata
         /// </summary>
         /// <returns>L'azione di mappatura</returns>
         public override Action<Input, Output> GetMapping() => new((In, Out) =>
         {
            // Numero di blocchi uguale alla lunghezza del vettore di rilevamenti diviso la somma delle celle
            var characteristics = In.detection.Length / ((80 * 80 + 40 * 40 + 20 * 20) * 3);
            // Il numero di classi del modello, dato dalla parte eccedente all'informazione di box piu' score all'interno dei blocchi
            var numClasses = characteristics - 5;
            // Lista di risultati
            var results = new List<(RectangleF Box, float Class, float Score, bool Valid)>();
            // Informazioni sul modello
            var info = In.D73BD0CB_FEA4_4EC9_9CD0_74B88DFF44F2.Split(new[] { '|' });
            var modelImageWidth = int.Parse(info[0]);
            var modelImageHeight = int.Parse(info[1]);
            var scoreConfidence = float.Parse(info[2], CultureInfo.InvariantCulture);
            var perCategoryConfidence = float.Parse(info[3], CultureInfo.InvariantCulture);
            var nmsOverlapRatio = float.Parse(info[4], CultureInfo.InvariantCulture);
            // Loop su tutti i blocchi di rilevamento
            for (int i = 0; i < In.detection.Length; i += characteristics) {
               // Filtra i box che hanno un punteggio inferiore alla soglia minima
               var objConf = In.detection[i + 4];
               if (objConf <= scoreConfidence)
                  continue;
               // Ottiene il punteggio reale delle classi e ne cerca quello con core massimo
               var maxConf = -float.MaxValue;
               var maxClass = -1;
               for (var j = 0; j < numClasses; j++) {
                  var score = In.detection[i + 5 + j] * objConf;
                  if (score > maxConf) {
                     maxConf = score;
                     maxClass = j;
                  }
               }
               // Verifica che lo score massimo sia superiore alla soglia minima
               if (maxConf < perCategoryConfidence)
                  continue;
               // Aggiunge il risultato
               var xc = In.detection[i + 1] / modelImageWidth;    // Centro x
               var yc = In.detection[i + 0] / modelImageHeight;   // Centro y
               var w = In.detection[i + 3] / modelImageWidth;     // Larghezza
               var h = In.detection[i + 2] / modelImageHeight;    // Altezza
               results.Add((Box: new RectangleF(xc - w / 2, yc - h / 2, w, h), Class: maxClass + 1, Score: maxConf, Valid: true));
            }
            // NMS. Elimina i box con sovrapposizione maggiore del parametro
            if (nmsOverlapRatio < 1f) {
               for (var i = 0; i < results.Count; i++) {
                  var item = results[i];
                  if (!item.Valid)
                     continue;
                  for (var j = 0; j < results.Count; j++) {
                     var current = results[j];
                     if (current == item || !current.Valid)
                        continue;
                     var intersection = RectangleF.Intersect(item.Box, current.Box);
                     var intArea = intersection.Width * intersection.Height;
                     var unionArea = item.Box.Width * item.Box.Height + current.Box.Width * current.Box.Height - intArea;
                     var overlap = intArea / unionArea;
                     if (overlap > nmsOverlapRatio) {
                        if (item.Score > current.Score) {
                           current.Valid = false;
                           results[j] = current;
                        }
                     }
                  }
               }
            }
            results = results.Where(item => item.Valid).ToList();
            // Scrive il risultato nelle colonne di uscita
            Out.detection_boxes = new float[results.Count * 4];
            Out.detection_classes = new float[results.Count];
            Out.detection_scores = new float[results.Count];
            for (var i = 0; i < results.Count; i++) {
               var (Box, Class, Score, Valid) = results[i];
               var iBox = i * 4;
               Out.detection_boxes[iBox + 0] = Box.Left;
               Out.detection_boxes[iBox + 1] = Box.Top;
               Out.detection_boxes[iBox + 2] = Box.Right;
               Out.detection_boxes[iBox + 3] = Box.Bottom;
               Out.detection_classes[i] = Class;
               Out.detection_scores[i] = Score;
            }
         });
         #endregion
      }
   }

   /// <summary>
   /// Transformer
   /// </summary>
   public sealed partial class Yolov5Transformer // Options
   {
      public sealed class Options
      {
         #region Properties
         /// <summary>
         /// Colonna di output dei riquadri
         /// </summary>
         public string BoxesColumnName { get; set; } = "detection_boxes";
         /// <summary>
         /// Colonna di output delle classi rilevate
         /// </summary>
         public string ClassesColumnName { get; set; } = "detection_classes";
         /// <summary>
         /// Colonne di input (dati che escono dal modello Yolov5)
         /// </summary>
         public string InputColumnName { get; set; } = "detection";
         /// <summary>
         /// Soglia minima del punteggio per categoria
         /// </summary>
         public float MinPerCategoryConfidence { get; set; } = 0.25f;
         /// <summary>
         /// Soglia minima del punteggio di rilevamento della singola cella
         /// </summary>
         public float MinScoreConfidence { get; set; } = 0.2f;
         /// <summary>
         /// Altezza dell'immagine del modello
         /// </summary>
         public int ModelImageHeight { get; set; } = 640;
         /// <summary>
         /// Larghezza dell'immagine del modello
         /// </summary>
         public int ModelImageWidth { get; set; } = 640;
         /// <summary>
         /// Rapporto di sovrapposizione minimo per la rimozione delle sovrapposizioni
         /// </summary>
         public float NmsOverlapRatio { get; set; } = 0.45f;
         /// <summary>
         /// Colonna di output dei punteggi rilevati
         /// </summary>
         public string ScoresColumnName { get; set; } = "detection_scores";
         #endregion
      }
   }
}