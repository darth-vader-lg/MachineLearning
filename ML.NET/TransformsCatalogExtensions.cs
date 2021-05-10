﻿using MachineLearning.Transforms;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using System;
using System.Reflection;

namespace Microsoft.ML
{
   /// <summary>
   /// Estensioni del catalogo di trasformazioni
   /// </summary>
   public static class TransformsCatalogExtensions
   {
      #region Methods
      /// <summary>
      /// Aggiunge una colonna con una costante
      /// </summary>
      /// <param name="catalog">Il catalogo di estensioni</param>
      /// <param name="value">Il valore costante da aggiungere</param>
      /// <param name="outputColumnName">Nome della colonna contenente il valore costante</param>
      /// <returns>L'Estimator</returns>
      /// <remarks>
      /// L'aggiunta della costante viene effettuata utilizzando l'Expression.
      /// Considerare il testo passato come facente parte della parte destra dell'assegnazione.
      /// </remarks>
      public static IEstimator<ITransformer> AddConst(this TransformsCatalog catalog, string value, string outputColumnName)
      {
         return
            catalog.CustomMapping(new Action<ConstMapper.Input, ConstMapper.Output>((In, Out) => { }), nameof(ConstMapper.Output.CE6C5270_FCDA_44F1_8680_7C5BF491B2B1))
            .Append(catalog.Expression(outputColumnName, $"i:{value}", nameof(ConstMapper.Output.CE6C5270_FCDA_44F1_8680_7C5BF491B2B1)))
            .Append(catalog.DropColumns(nameof(ConstMapper.Output.CE6C5270_FCDA_44F1_8680_7C5BF491B2B1)));
      }
      /// <summary>
      /// Restituisce l'interfaccia IChannelProvider di un catalogo
      /// </summary>
      /// <param name="catalog"></param>
      /// <returns>L'interfaccia</returns>
      internal static IChannelProvider GetChannelProvider(this TransformsCatalog catalog)
      {
         var prop = catalog.GetType().GetProperty("Microsoft.ML.Data.IInternalCatalog.Environment", BindingFlags.Instance | BindingFlags.NonPublic);
         return prop.GetMethod.Invoke(catalog, null) as IChannelProvider;
      }
      /// <summary>
      /// Esegue lo scoring dell'output di un modello Yolov5
      /// </summary>
      /// <param name="catalog">Catalogo di trasformazioni</param>
      /// <param name="inputColumnName">Nome della colonna di input (la colonna di output del modello)</param>
      /// <param name="classesColumnName">Il nome della colonna di output contenente le classi di rilevamento</param>
      /// <param name="scoresColumnName">Il nome della colonna di output contenente i punteggi di rilevamento</param>
      /// <param name="boxesColumnName">Il nome della colonna di output contenente i gruppi di x1,y1,x2,y2 dei bounding box in coordinate da 0 a 1</param>
      /// <param name="modelImageWidth">Larghezza immagine del modello</param>
      /// <param name="modelImageHeight">Altezza immagine del modello</param>
      /// <param name="minScoreConfidence">Minimo punteggio previsione</param>
      /// <param name="minPerCategoryConfidence">Minimo punteggio previsione per categoria</param>
      /// <param name="nmsOverlapRatio">Fattore di sovrapposizione NMS</param>
      /// <returns></returns>
      public static Yolov5Estimator ScoreYolov5(
         this TransformsCatalog catalog,
         string inputColumnName = "detection",
         string classesColumnName = "detection_classes",
         string scoresColumnName = "detection_scores",
         string boxesColumnName = "detection_boxes",
         int modelImageWidth = 640,
         int modelImageHeight = 640,
         float minScoreConfidence = 0.2f,
         float minPerCategoryConfidence = 0.25f,
         float nmsOverlapRatio = 0.45f) => ScoreYolov5(
            catalog,
            new Yolov5Transformer.Options
            {
               BoxesColumnName = boxesColumnName,
               ClassesColumnName = classesColumnName,
               InputColumnName = inputColumnName,
               ModelImageHeight = modelImageHeight,
               ModelImageWidth = modelImageWidth,
               NmsOverlapRatio = nmsOverlapRatio,
               MinPerCategoryConfidence = minPerCategoryConfidence,
               MinScoreConfidence = minScoreConfidence,
               ScoresColumnName = scoresColumnName
            });
      /// <summary>
      /// Esegue lo scoring dell'output di un modello Yolov5
      /// </summary>
      /// <param name="catalog">Catalogo di trasformazioni</param>
      /// <param name="options">Opzioni</param>
      /// <returns>L'estimatore</returns>
      public static Yolov5Estimator ScoreYolov5(this TransformsCatalog catalog, Yolov5Transformer.Options options) => new(catalog, options);
      #endregion
   }

   /// <summary>
   /// Mapper per l'assegnazione delle costanti
   /// </summary>
   [CustomMappingFactoryAttribute(nameof(Output.CE6C5270_FCDA_44F1_8680_7C5BF491B2B1))]
   internal class ConstMapper : CustomMappingFactory<ConstMapper.Input, ConstMapper.Output>
   {
      #region Properties
      /// <summary>
      /// Dati di input (vuoti)
      /// </summary>
      internal class Input { }
      /// <summary>
      /// Dati di output (solo una stringa come placeholder)
      /// </summary>
      internal class Output { public string CE6C5270_FCDA_44F1_8680_7C5BF491B2B1 { get; set; } }
      #endregion
      #region Methods
      /// <summary>
      /// Azione di mappatura
      /// </summary>
      /// <returns></returns>
      public override Action<Input, Output> GetMapping() => new((input, output) => { });
      #endregion
   }
}
